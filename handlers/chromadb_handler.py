import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("test")

from datasets import load_dataset

dataset = load_dataset("sciq", split="train")

# Filter the dataset to only include questions with a support
dataset = dataset.filter(lambda x: x["support"] != "")

print("Number of questions with support: ", len(dataset))


collection = client.create_collection("sciq_supports")

from tqdm.notebook import tqdm

# Load the supporting evidence in batches of 1000
batch_size = 1000
for i in tqdm(range(0, len(dataset), batch_size), desc="Adding documents"):
    collection.add(
        ids=[
            str(i) for i in range(i, min(i + batch_size, len(dataset)))
        ],  # IDs are just strings
        documents=dataset["support"][i : i + batch_size],
        metadatas=[
            {"type": "support"} for _ in range(i, min(i + batch_size, len(dataset)))
        ],
    )