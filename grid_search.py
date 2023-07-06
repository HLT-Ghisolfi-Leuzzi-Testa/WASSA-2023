def optuna_hp_space(trial):
    return {
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]), # TODO: to decide
    }

best_trial = trainer.hyperparameter_search(
  direction="maximize",
  backend="optuna",
  hp_space=optuna_hp_space,
  n_trials=1
)

train_arguments = TrainingArguments(
    output_dir="./",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    seed=SEED
) # TODO: custom other params (fixed ones)

def model_init():
  model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification")
  model.to(device)
  return model

trainer = Trainer(
    model_init=model_init,
    args=train_arguments,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    compute_metrics=compute_EMO_metrics_trainer
)