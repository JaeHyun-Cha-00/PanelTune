from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

def clamp(settings, learning_rate_limits=(1e-5, 5e-5), epochs_limits=(1, 3)):
    # keep the builder's settings inside the limits (from the config); missing ones use the defaults
    learning_rate = float(settings.get("learning_rate", 2e-5))
    epochs = int(settings.get("epochs", 1))
    low, high = [float(x) for x in learning_rate_limits]
    first, last = [int(x) for x in epochs_limits]
    return {
        "learning_rate": min(max(learning_rate, low), high),
        "epochs": min(max(epochs, first), last),
    }

def train(model_path, examples, output_dir, learning_rate, epochs):
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = Dataset.from_list(examples)

    config = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
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