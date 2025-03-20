from tqdm import tqdm
import modal
import modal.runner
import pandas as pd
import torch
import torch.nn.functional as F

import numpy as np

app = modal.App.lookup(name="recommender", create_if_missing=True)
volume = modal.Volume.from_name("recommender", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.9").pip_install("pandas", "torch","tqdm","sentence-transformers","numpy")

### --- BATCH SIZE: TASK 3
# I imagine a larger batch size will use fewer cpus/gpus and memory. A smaller one will run more computations in parallel but use more resources.
# Not really a point in lowering batch size since we limited cpu usage so heavily in task 1. Changing to 3. 

# --- Batch size 1: GPU VRAM: 1.7 GiB, RAM: 1023 MiB, Excec time: 1m 23s 
# --- Batch size 8: GPU VRAM: 3.03 GiB, RAM: 556 MiB , Excec time: 1m 26s
# --- Batch size 12: GPU VRAM: 3.31 GiB, RAM: 558 MiB, Excec time: 1m 16s
# --- Batch size 20: GPU VRAM: 3.06 GiB, RAM: 485 MiB , Excec time: 1m 13s
# --- Batch size 30: GPU VRAM: 2.9 GiB, RAM: 477 MiB, Excec time: 1m 17s
# The batch size of 20 seems to be the best. It uses the least amount of resources and has the fastest execution time.

def batch_infer(df, inputs, batch_size=20, device="cuda"):
    embed_matrix = [df.loc[df.title.isin(movies)].embed.tolist() for movies in inputs]
    embed_matrix = torch.tensor(embed_matrix).to(device)
    embeds = embed_matrix.mean(dim=1)
    rest_embeds = df.loc[~df.title.isin(inputs[0])].embed.tolist()
    rest_embeds = torch.tensor(rest_embeds).to(device)
    sim = []
    for i in tqdm(range(0, len(embeds), batch_size)):
        batch_similarity = batch_similarity = F.cosine_similarity(
            embeds[i : i + batch_size].unsqueeze(1),
            rest_embeds.unsqueeze(0),
            dim=2,
        )
        sim.append(batch_similarity)
    sim = torch.cat(sim)
    idx = torch.topk(sim, 5).indices.cpu().numpy()
    recommendations = []
    for ix, movies in enumerate(inputs):
        recs = df.loc[~df.title.isin(movies)].iloc[idx[ix]].title.tolist()
        recommendations.append(recs)
    return recommendations

def upload_data():
    with volume.batch_upload() as batch:
        batch.put_file("Week_6/movies_embeds.pkl", "/data/movies_embeds.pkl")
        batch.put_file("Week_6/infer_dataset_10k.csv", "/data/infer_dataset_10k.csv")

### --- LIMITING USAGE: TASK 1
# Noticed there was a big spike in CPU in the beginning then a drop off to around 1.02 cpus. Would be nice to limit it to 2 cpus. 
# Also, the peak mem usage is around 601 MiB. Good to limit to 2000 in case of memory leak. 

### --- UPDATE: Everything went great. CPU limited to 1 core and memory limited to 2000 MiB for safety. Took longer to run.

cpu_request = 0.125 #default 
cpu_limit = 3 # TASK 3: Changed to 3 to fully utilize the newer beefier GPU

mem_request = 1024 #default
mem_limit = 2000


### --- SHEDULING FOR MON, WED, FRI: TASK 2
# The app is scheduled to run at 9am on Monday, Wednesday, and Friday.

@app.function(
    cpu=(cpu_request, cpu_limit), 
    memory=(mem_request, mem_limit),
    schedule=modal.Cron("0 9 * * 1,3,5"),
    volumes={"/volume": volume},
    image=image,
    gpu=["L40S"], ### --- BIGGER DATASET: TASK 3: BEEFY ASS GPU WITH TONS OF MEMORY SHOULD DO THE TRICK
)
def infer():
    volume.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    movie_database = pd.read_pickle("/volume/data/movies_embeds.pkl")
    daily_dataset = pd.read_csv("/volume/data/infer_dataset_10k.csv")
    recommendations = batch_infer(movie_database, daily_dataset.values, device=device)
    recommendations = pd.DataFrame(
        recommendations, columns=["rec1", "rec2", "rec3", "rec4", "rec5"]
    )
    recommendations.to_csv("/volume/data/daily_recommendations_100k.csv", index=False)


if __name__ == "__main__":
    upload_data()
    modal.runner.deploy_app(app)


# Just copied this over for the 10k dataset which worked fine this code works bro trust
# If run this just remember to put the 10k dataset into the modal volume bro 