import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval import evaluator, tasks

class PipelineLM(LM):
    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__()
        self._model = model
        self.tokenizer = tokenizer
        self._device = device
        
    def loglikelihood(self, requests):
        res = []
        for req in requests:
            context, continuation = req.args
            ctx_tokens = self.tokenizer.encode(context).ids
            cont_tokens = self.tokenizer.encode(continuation).ids
            
            if len(cont_tokens) == 0:
                res.append((0.0, False))
                continue
                
            inps = torch.tensor([ctx_tokens + cont_tokens], dtype=torch.long).to(self.device)
            target_ids = inps[0, len(ctx_tokens):]
            
            with torch.no_grad():
                logits, _ = self._model(inps)
            
            cont_logits = logits[0, len(ctx_tokens)-1 : len(ctx_tokens)-1+len(cont_tokens), :]
            logprobs = F.log_softmax(cont_logits, dim=-1)
            
            ll = sum([logprobs[i, target_ids[i]].item() for i in range(len(cont_tokens))])
            is_greedy = (logprobs.argmax(-1) == target_ids).all().item()
            res.append((ll, is_greedy))
        return res
        
    def loglikelihood_rolling(self, requests):
        return [(-1.0,) for _ in requests]
        
    def generate_until(self, requests):
        return [""] * len(requests)
        
    @property
    def eot_token_id(self):
        return self.tokenizer.encode("<|endoftext|>").ids[0]

    @property
    def max_length(self):
        return self._model.config.block_size
        
    @property
    def max_gen_toks(self):
        return 256
        
    @property
    def batch_size(self):
        return 1
        
    @property
    def device(self):
        return self._device

class FastEvaluator:
    def __init__(self, tasks_list=["hellaswag"]):
        self.tasks_list = tasks_list
        # Disable huggingface datasets progress bars so they don't spam the console if they do re-run
        import datasets
        datasets.disable_progress_bar()
        
        # Load and map the tasks exactly once
        self.task_manager = tasks.TaskManager()
        self.task_dict = tasks.get_task_dict(self.tasks_list, task_manager=self.task_manager)
        
    def evaluate(self, model, tokenizer, device, limit=None):
        lm = PipelineLM(model, tokenizer, device)
        results = evaluator.evaluate(
            lm=lm,
            task_dict=self.task_dict,
            limit=limit,
        )
        if results is not None and "results" in results:
            task_name = self.tasks_list[0]
            if task_name in results["results"]:
                return results["results"][task_name].get("acc,none", 0.0)
        return 0.0
