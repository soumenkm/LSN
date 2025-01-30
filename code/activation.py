import json, os, sys, tqdm, pickle, datetime, random
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
torch.manual_seed(42)
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Union
import pandas as pd
from dataset import WikipediaDatasetHF
from models import ModelForMLM
from utils import lang_map, models_map

class NeuronRelevance:
    def __init__(self, device: torch.device, model_name: str, quant_config: Union[None, BitsAndBytesConfig], lang: str, scoring_method: str):
        """scoring_method: Any["all_act", "act_stat"]
        """
        self.cwd = Path.cwd()
        self.device = device
        self.lang = lang
        self.model_name = model_name
        self.model_name_srt = self.model_name.split('/')[-1]
        self.method = scoring_method
        self.method_list = ["act_abs_mean", "act_prob_mean", "act_prob_zero", "act_prob_95p", "act_prob_90p", "act_prob_75p"]
        if scoring_method == "act_stat":
            self.rel_data_path = Path(self.cwd, f"outputs/activation/{self.model_name_srt}/{self.method}/rel_{self.lang}.pkl")
        else:
            self.rel_data_path = Path(self.cwd, f"outputs/activation/{self.model_name_srt}/{self.method}/rel_{self.lang}.pkl")           
        Path.mkdir(self.rel_data_path.parent, exist_ok=True, parents=True)
        
        if not self.rel_data_path.exists():
            self.quant_config = quant_config
            self.model = ModelForMLM(device=self.device, model_name=self.model_name, quant_config=self.quant_config)
            self.dataset = WikipediaDatasetHF(model_name=self.model_name, lang=self.lang, max_context_len=512)
        
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_act_stat_data(self, batch_size: Union[int, None], data_frac: Union[float, None]) -> dict:
        if self.rel_data_path.exists():
            out_obj = pickle.load(open(self.rel_data_path, "rb"))
            print(f"{self.info()}: The relevance {self.method} data is loaded from {self.rel_data_path}")
            return out_obj
        
        ds = Subset(self.dataset, indices=random.sample(range(len(self.dataset)), k=int(len(self.dataset)*data_frac)))
        dl = DataLoader(dataset=ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
        
        mean_mu_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_std_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p50_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p75_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p90_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p95_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p5_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p10_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        mean_p25_tensor = torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device)
        N = 0
        with tqdm.tqdm(iterable=dl, desc=f"Computing activation for lang: {self.lang}", unit="batches", colour="green") as pbar:
            for input_dict in pbar:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(**input_dict)
            
                mu_list = []
                std_list = []
                p90_list = []
                p95_list = []
                p75_list = []
                p50_list = []
                p25_list = []
                p10_list = []
                p5_list = []
                
                for layer_idx in range(len(self.model.activations.keys())):
                    act = self.model.activations[layer_idx] # (b, T, 4d)
                    mu = act.mean(dim=(0,1)) # (4d,)
                    mu_list.append(mu.clone().detach()) # (L, 4d)
                    std_list.append(act.std(dim=(0,1)).clone().detach()) # (L, 4d)
                    p50_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.50, dim=0).clone().detach()) # (L, 4d)
                    p75_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.75, dim=0).clone().detach()) # (L, 4d)
                    p90_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.90, dim=0).clone().detach()) # (L, 4d)
                    p95_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.95, dim=0).clone().detach()) # (L, 4d)
                    p5_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.05, dim=0).clone().detach()) # (L, 4d)
                    p10_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.10, dim=0).clone().detach()) # (L, 4d)
                    p25_list.append(torch.quantile(act.flatten(0,1).to(torch.float32), q=0.25, dim=0).clone().detach()) # (L, 4d)
                    
                mean_mu_tensor += torch.stack(mu_list, dim=0) # (L, 4d)
                mean_std_tensor += torch.stack(std_list, dim=0) # (L, 4d)
                mean_p50_tensor += torch.stack(p50_list, dim=0) # (L, 4d)
                mean_p75_tensor += torch.stack(p75_list, dim=0) # (L, 4d)
                mean_p90_tensor += torch.stack(p90_list, dim=0) # (L, 4d)
                mean_p95_tensor += torch.stack(p95_list, dim=0) # (L, 4d)
                mean_p5_tensor += torch.stack(p5_list, dim=0) # (L, 4d)
                mean_p10_tensor += torch.stack(p10_list, dim=0) # (L, 4d)
                mean_p25_tensor += torch.stack(p25_list, dim=0) # (L, 4d)
                N += 1
    
        mean_mu_tensor = mean_mu_tensor/N # (L, 4d)
        mean_std_tensor = mean_std_tensor/N # (L, 4d)
        mean_p50_tensor = mean_p50_tensor/N # (L, 4d)
        mean_p75_tensor = mean_p75_tensor/N # (L, 4d)
        mean_p90_tensor = mean_p90_tensor/N # (L, 4d)
        mean_p95_tensor = mean_p95_tensor/N # (L, 4d)
        mean_p5_tensor = mean_p5_tensor/N # (L, 4d)
        mean_p10_tensor = mean_p10_tensor/N # (L, 4d)
        mean_p25_tensor = mean_p25_tensor/N # (L, 4d)
        
        data = {
            "lang": self.lang,
            "mean_mu_act": mean_mu_tensor.cpu(),
            "mean_std_act": mean_std_tensor.cpu(),
            "mean_p50_act": mean_p50_tensor.cpu(),
            "mean_p75_act": mean_p75_tensor.cpu(),
            "mean_p90_act": mean_p90_tensor.cpu(),
            "mean_p95_act": mean_p95_tensor.cpu(),
            "mean_p5_act": mean_p5_tensor.cpu(),
            "mean_p10_act": mean_p10_tensor.cpu(),
            "mean_p25_act": mean_p25_tensor.cpu(),
        }
        pickle.dump(data, open(self.rel_data_path, "wb"))
        print(f"{self.info()}: The relevance {self.method} data is stored at {self.rel_data_path}")
        return data

    def get_relevance_data(self, batch_size: Union[int, None], data_frac: Union[float, None]) -> dict:
        if self.rel_data_path.exists():
            out_obj = pickle.load(open(self.rel_data_path, "rb"))
            print(f"{self.info()}: The relevance data {self.method} is loaded from {self.rel_data_path}")
            return out_obj
        
        ds = Subset(self.dataset, indices=random.sample(range(len(self.dataset)), k=int(len(self.dataset)*data_frac)))
        dl = DataLoader(dataset=ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
        
        act_stat_path = Path(Path.cwd(), f"outputs/activation/{self.model_name_srt}/act_stat/rel_{self.lang}.pkl")
        if act_stat_path.exists():
            act_stat_data = pickle.load(open(act_stat_path, "rb"))
            print(f"The activation stat data is loaded from {act_stat_path}")
        else:
            raise ValueError(f"{act_stat_path} doesn't exist!")
        
        mean_rel_tensor = {k: torch.zeros(size=(self.model.L, self.model.int_d)).to(self.device) for k in self.method_list}
        N = 0
        with tqdm.tqdm(iterable=dl, desc=f"Computing activation for lang: {self.lang}", unit="batches", colour="green") as pbar:
            for input_dict in pbar:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(**input_dict)
            
                theta_dict = {k: [] for k in self.method_list}
                rel = {}
                theta = {}
                for layer_idx in range(len(self.model.activations.keys())):
                    act = self.model.activations[layer_idx] # (b, T, 4d)
                    rel["act_abs_mean"] = torch.abs(act) # (b, T, 4d)
                    theta["act_abs_mean"] = rel["act_abs_mean"].mean(dim=(0,1)) # (4d,)
                    rel["act_prob_zero"] = (act > 0).to(torch.float16) # (b, T, 4d)
                    theta["act_prob_zero"] = rel["act_prob_zero"].mean(dim=(0,1)) # (4d,)
                    rel["act_prob_mean"] = (act > act_stat_data["mean_mu_act"][layer_idx].to(self.device)).to(torch.float16) # (b, T, 4d)
                    theta["act_prob_mean"] = rel["act_abs_mean"].mean(dim=(0,1)) # (4d,)
                    rel["act_prob_95p"] = (act > act_stat_data["mean_p95_act"][layer_idx].to(self.device)).to(torch.float16) # (b, T, 4d)
                    theta["act_prob_95p"] = rel["act_prob_95p"].mean(dim=(0,1)) # (4d,)
                    rel["act_prob_90p"] = (act > act_stat_data["mean_p90_act"][layer_idx].to(self.device)).to(torch.float16) # (b, T, 4d)
                    theta["act_prob_90p"] = rel["act_prob_90p"].mean(dim=(0,1)) # (4d,)
                    rel["act_prob_75p"] = (act > act_stat_data["mean_p75_act"][layer_idx].to(self.device)).to(torch.float16) # (b, T, 4d)
                    theta["act_prob_75p"] = rel["act_prob_75p"].mean(dim=(0,1)) # (4d,)
                    
                    for key in self.method_list:
                        theta_dict[key].append(theta[key].clone().detach()) # (L, 4d)
                
                for key in self.method_list:
                    mean_rel_tensor[key] += torch.stack(theta_dict[key], dim=0) # (L, 4d)          
                N += 1

        for key in self.method_list:
            mean_rel_tensor[key] = mean_rel_tensor[key]/N # (L, 4d)
            mean_rel_tensor[key] = mean_rel_tensor[key].cpu()
        data = {
            "lang": self.lang,
            "mean_rel": mean_rel_tensor,
        }
        pickle.dump(data, open(self.rel_data_path, "wb"))
        print(f"{self.info()}: The relevance {self.method} data is stored at {self.rel_data_path}")
        return data

       
def main(model_name: str, device: torch.device) -> None:
    methods = ["all_act", "act_stat"]
    quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',  
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,  
    )
    for lang in ["bn"]: 
        for method in ["all_act"]:
            rel = NeuronRelevance(device=device, model_name=model_name, quant_config=quant_config, lang=lang, scoring_method=method)
            if method == "act_stat":
                out = rel.get_act_stat_data(batch_size=4, data_frac=0.1)
            else:
                out = rel.get_relevance_data(batch_size=4, data_frac=0.2)
            print(out) 
    print("DONE")
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["mistral-nemo"], device=device)
    
    