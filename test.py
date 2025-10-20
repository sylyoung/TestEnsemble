import os

# --- Configuration ---
folder = "/Users/Riccardo/Workspace/Github/TestEnsemble/logs/helm/raft"  # change this to your folder path

keep_substrings = [
    "meta_llama-2-70b",
    "openai_gpt-3.5-turbo-0301",
    "ai21_j2-jumbo",
    "writer_palmyra-x",
    "tiiuae_falcon-40b",
    "cohere_command-xlarge-beta",
    "microsoft_TNLGv2_530B",
    "mosaicml_mpt-instruct-30b",
    "anthropic_stanford-online-all-v4-s3",
    "together_redpajama-incite-instruct-7b",
]

# --- Execution ---
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if not os.path.isfile(filepath):
        continue  # skip subfolders

    if not any(sub in filename for sub in keep_substrings):
        print(f"Deleting: {filename}")
        os.remove(filepath)
