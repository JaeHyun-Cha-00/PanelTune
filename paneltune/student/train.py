from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

def train(model_path, examples, output_dir):
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = Dataset.from_list(examples)

    config = SFTConfig(
        output_dir=output_dir,
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir

if __name__ == "__main__":
    pass   