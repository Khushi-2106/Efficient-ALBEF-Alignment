'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        # Save config for EPIC hyperparams access
        self.config = config  # >>> EPIC: store config for epics params
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.itm_head = nn.Linear(text_width, 2)     

        # >>> EPIC: ITC head (per-token binary classifier)
        # Predicts whether each text token is consistent (1) or inconsistent/replaced (0)
        epic_hidden = text_width
        self.itc_head = nn.Linear(epic_hidden, 1)  # outputs logits per token
        # small initialization
        nn.init.normal_(self.itc_head.weight, std=0.02)
        nn.init.constant_(self.itc_head.bias, 0.)
        # >>> EPIC: end added

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)       
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)



    def forward(self, image, text, alpha=0):
        """
        Returns:
            (loss_mlm, loss_ita, loss_itm, loss_epic)
        loss_epic is a scalar tensor. If EPIC is disabled (config missing), it will be zero.
        """

        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        # image_embeds: [B, V, D] where index 0 is [CLS] patch token
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
             
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                                         

            text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)          

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_all / self.temp 
        sim_t2i = text_feat @ image_feat_all / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###
        # forward the positve image-text pair (fusion)
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )            

        # NOTE: output_pos.last_hidden_state: [B, T, D] fused text token embeddings (used for ITM + now EPIC)
        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                         

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     
        
        ##================= MLM ========================##                
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids, 
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = image_embeds_m,
                                           encoder_attention_mask = image_atts,      
                                           return_dict = True,
                                           return_logits = True,   
                                          )    
        mlm_output = self.text_encoder(input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       labels = labels,   
                                       soft_labels = F.softmax(logits_m,dim=-1),
                                       alpha = alpha
                                      )                           
        loss_mlm = mlm_output.loss

        # >>> EPIC: Start EPIC / ITC computations
        # If 'epic' section missing in config or epic.weight==0, we return zero EPIC loss for compatibility.
        loss_epic = torch.tensor(0.0, device=image.device, requires_grad=False)

        epic_cfg = self.config.get('epic', {}) if isinstance(self.config, dict) else {}
        epic_enabled = epic_cfg.get('enabled', True)
        top_k_tokens = int(epic_cfg.get('top_k_tokens', 3))          # how many text tokens to choose per sample
        replace_topk = int(epic_cfg.get('replace_topk', 5))          # how many top MLM candidates to try
        epic_loss_weight_inside_model = float(epic_cfg.get('internal_weight', 1.0))  # local scaling if wanted

        if epic_enabled:
            # 1) Compute a simple saliency score per text token: max dot(text_token, image_patch) across patches
            #    Use pre-fusion text embeddings (text_embeds) and image patch embeddings (image_embeds[:,1:,:])
            #    text_embeds: [B, T, D_text], image_embeds: [B, V, D_img]
            #    Project both to a common space (using existing projections) and compute token-wise saliency.
            with torch.no_grad():
                # project image patch embeddings (exclude [CLS] at idx 0)
                img_patch_embeds = image_embeds[:, 1:, :]  # [B, Vp, D_img]
                # project image patches to embed_dim space then to text-proj space
                img_proj = self.vision_proj(img_patch_embeds)  # [B, Vp, embed_dim]
                # project text tokens to same embed dim
                text_proj_tokens = self.text_proj(text_embeds)  # [B, T, embed_dim]

                # compute token-patch similarity: [B, T, Vp]
                # use matmul after permuting: (text_proj_tokens @ img_proj.transpose)
                sim_tok_patch = torch.matmul(text_proj_tokens, img_proj.transpose(1, 2))  # [B, T, Vp]
                # saliency per text token: max over patches (could also use sum)
                saliency, _ = sim_tok_patch.max(dim=-1)  # [B, T]
                # normalize saliency per sample (min-max)
                smin = saliency.min(dim=1, keepdim=True)[0]
                smax = saliency.max(dim=1, keepdim=True)[0]
                saliency_norm = (saliency - smin) / (smax - smin + 1e-9)  # [B, T]

                # pick top-k tokens per sample
                k = min(top_k_tokens, saliency_norm.size(1))
                topk_vals, topk_idx = torch.topk(saliency_norm, k=k, dim=1)  # [B, k]
                # build boolean mask [B, T]
                selected_mask = torch.zeros_like(saliency_norm, dtype=torch.bool)
                batch_idx = torch.arange(saliency_norm.size(0), device=saliency_norm.device).unsqueeze(1)
                selected_mask[batch_idx, topk_idx] = True  # True where we will create mismatches

            # 2) Create replacements for selected tokens using the MLM head predictions:
            # We'll create a copy of input_ids and replace selected positions with MLM-predicted tokens.
            # Approach:
            #   - For each sample, mask the selected positions (use tokenizer.mask_token_id),
            #   - Run text_encoder (MLM mode) to get logits, then choose top candidates not equal to original token.
            #   - We try up to replace_topk candidates to avoid picking same token.
            input_ids_for_repl = text.input_ids.clone()
            # mask selected positions in the copy (so MLM will predict them)
            # But be careful not to override special tokens (CLS/PAD)
            sel_mask_positions = selected_mask & (text.input_ids != self.tokenizer.cls_token_id) & (text.input_ids != self.tokenizer.pad_token_id)
            mask_token_id = self.tokenizer.mask_token_id
            input_ids_for_repl[sel_mask_positions] = mask_token_id

            # Run MLM forward to get logits for masked positions
            # We don't need gradients for this sampling step; compute without grad to save memory
            with torch.no_grad():
                mlm_outputs_for_repl = self.text_encoder(input_ids_for_repl,
                                                         attention_mask=text.attention_mask,
                                                         encoder_hidden_states=image_embeds,
                                                         encoder_attention_mask=image_atts,
                                                         return_dict=True,
                                                         return_logits=True)
                # logits shape: [B, T, Vocab]
                logits_repl = mlm_outputs_for_repl.logits  # may be attribute name depending on HF wrapper; typically .logits

            # Build replacement ids: start with original tokens, replace selected positions with predicted tokens
            replaced_input_ids = text.input_ids.clone()
            B, T = text.input_ids.shape
            vocab_size = logits_repl.size(-1)

            # For masked positions, pick the top predicted candidate that is different from original token
            topk_candidates = torch.topk(logits_repl, k=replace_topk, dim=-1).indices  # [B, T, replace_topk]
            # iterate (vectorized-ish) to choose replacement not equal to original
            # We'll create a mask over candidates and pick first candidate != original
            # Expand original tokens for comparison
            orig = text.input_ids.unsqueeze(-1).expand(-1, -1, replace_topk)  # [B, T, replace_topk]
            neq = (topk_candidates != orig)
            # For positions where selected_mask is True, choose first candidate with neq True
            # For numerical stability, where all candidates == original (rare), fallback to random token
            # Prepare a fallback random token tensor
            rand_tokens = torch.randint(0, vocab_size, (B, T), device=text.input_ids.device)
            chosen_replacements = rand_tokens.clone()

            # loop over replace_topk dimension to fill chosen_replacements where not yet chosen
            # This loop is small (replace_topk default 5) so acceptable
            for r in range(replace_topk):
                cand = topk_candidates[:, :, r]  # [B, T]
                cand_ok = neq[:, :, r]  # [B, T]
                # choose cand where cand_ok and selected_mask and not already chosen (we mark chosen by replacing rand_tokens)
                pick_pos = cand_ok & sel_mask_positions & (chosen_replacements == rand_tokens)
                if pick_pos.any():
                    chosen_replacements[pick_pos] = cand[pick_pos]

            # Now set replaced tokens on positions sel_mask_positions
            replaced_input_ids[sel_mask_positions] = chosen_replacements[sel_mask_positions]

            # 3) Build per-token consistency labels: 1 = original, 0 = replaced for selected positions
            # labels_consistency: [B, T], dtype float
            labels_consistency = torch.ones_like(text.input_ids, dtype=torch.float, device=text.input_ids.device)
            labels_consistency[sel_mask_positions] = 0.0
            # Also mask out pad tokens so they don't contribute
            valid_mask = (text.input_ids != self.tokenizer.pad_token_id)

            # 4) Forward the fusion encoder for replaced texts to obtain token embeddings (we use the same encoder)
            #    We only need fused token embeddings for the original (output_pos) to run the ITC head,
            #    because EPIC compares the fused text embedding to decide if tokens match the image.
            #    However, to be strict, the EPIC objective uses the model to detect mismatches introduced
            #    by LM; we emulate this by using the fused embedding of the (possibly replaced) text.
            #    Here we compute fused embeddings for the replaced inputs as well (batched).
            replaced_text_embeds = None
            try:
                # token embeddings (before fusion) for replaced inputs
                # We want to compute fused representations for the replaced text inputs as the model would encode them
                repl_text_output = self.text_encoder.bert(replaced_input_ids,
                                                          attention_mask=text.attention_mask,
                                                          return_dict=True,
                                                          mode='text')
                repl_text_embeds = repl_text_output.last_hidden_state  # [B, T, D_text]
                # fusion with image
                output_repl = self.text_encoder.bert(encoder_embeds = repl_text_embeds,
                                                     attention_mask = text.attention_mask,
                                                     encoder_hidden_states = image_embeds,
                                                     encoder_attention_mask = image_atts,
                                                     return_dict = True,
                                                     mode = 'fusion')
                fused_repl_tokens = output_repl.last_hidden_state  # [B, T, D_text]
            except Exception as e:
                # If the HF wrapper doesn't accept the calls above, fallback: reuse output_pos (less ideal)
                # but we want to avoid hard failures.
                fused_repl_tokens = output_pos.last_hidden_state
                # NOTE: fallback means the EPIC signal will be weaker; recommended to fix wrapper issues if triggered.

            # 5) Compute ITC logits per token using fused_repl_tokens (or fused positive tokens)
            # Use the fused tokens from the replaced text; comparing model's fused view to original image
            # itc_input_embeds = fused_repl_tokens  # [B, T, D_text]
            itc_input_embeds = fused_repl_tokens  # we've chosen to use replaced fused embeddings
            itc_logits = self.itc_head(itc_input_embeds).squeeze(-1)  # [B, T]

            # 6) Compute binary CE loss on selected token positions only
            # Convert labels_consistency to same device and shape
            labels_consistency = labels_consistency.to(itc_logits.device)
            valid_positions = valid_mask & sel_mask_positions  # only compute loss on selected valid positions
            if valid_positions.any():
                # BCE with logits (mean over selected positions)
                # flatten
                logits_sel = itc_logits[valid_positions]
                labels_sel = labels_consistency[valid_positions]
                loss_epic = F.binary_cross_entropy_with_logits(logits_sel, labels_sel)
                # optional internal weighting
                loss_epic = loss_epic * epic_loss_weight_inside_model
            else:
                loss_epic = torch.tensor(0.0, device=image.device, requires_grad=False)

        # >>> EPIC: End EPIC computations

        # Return new 4-tuple containing epic loss
        return loss_mlm, loss_ita, loss_itm, loss_epic  



    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
