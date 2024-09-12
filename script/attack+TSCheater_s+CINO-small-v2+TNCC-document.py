import sys
import os

sys.path.append(os.getcwd())

import OpenAttack
import open_attack
from datasets import load_dataset


def dataset_mapping(data):
    return {
        "x": data["text"],
        "y": data["label"]
    }


def main():
    print("Loading Attacker...")
    attacker = open_attack.attackers.TSCheater_s(lang="tibetan")

    print("Loading Victim ...")
    victim = OpenAttack.loadVictim("XLM-RoBERTa.CINO-small-v2+TNCC-document")

    print("Loading Dataset ...")
    dataset = load_dataset(path=os.path.join(os.getcwd(), "data", "Dataset.Loader", "TNCC-document.py"), split="test",
                           trust_remote_code=True).map(function=dataset_mapping)

    print("Start Attack!")
    attack_eval = open_attack.AttackEval(attacker, victim, "tibetan", metrics=[])
    attack_eval.eval(dataset, visualize=True, progress_bar=True, log_dir="Adv.TSCheater_s.CINO-small-v2+TNCC-document")


if __name__ == "__main__":
    main()