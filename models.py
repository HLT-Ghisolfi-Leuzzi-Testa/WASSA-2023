class BertPoolingByLexica(BertPreTrainedModel):
  def __init__(self, config, n_features, dropout=0.3):
    super().__init__(config)
    print(config)
    self.num_labels = config.num_labels
    self.config = config
    self.n_features = n_features
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(config.hidden_size+self.n_features, config.num_labels)
    self.post_init()

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    features=None,
    empathy_values=None,
    distress_values=None,
    emotion_values=None,
  ):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    # --------------------------------------------------------------------------
    mask = attention_mask.detach().clone()
    # unmask cls token
    mask[:,1] = 1.0
    # unmask tokens with high or low empathy, high distress levels, or expressing at least one emotion
    if empathy_values is not None:
      mask[(empathy_values>5) | (empathy_values<3)] = 1.0
    if distress_values is not None:
      mask[distress_values>4] = 1.0
    if emotion_values is not None:
      mask[emotion_values.sum(dim=-2)>=1] = 1.0
    # mean pooling of unmasked tokens
    output = outputs.last_hidden_state
    input_mask_expanded = mask.unsqueeze(-1).expand(output.size()).float()
    sum_embeddings = torch.sum(output * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min = 1e-9)
    pooled_output = sum_embeddings/sum_mask
    # --------------------------------------------------------------------------

    pooled_output = self.dropout(pooled_output)
    if features is not None: # concat global features
      pooled_output = torch.cat((pooled_output, features), dim=1)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
      if self.config.problem_type is None:
        if self.num_labels == 1:
            self.config.problem_type = "regression"
        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            self.config.problem_type = "single_label_classification"
        else:
            self.config.problem_type = "multi_label_classification"

      if self.config.problem_type == "regression":
        loss_fct = nn.MSELoss()
        if self.num_labels == 1:
          loss = loss_fct(logits.squeeze(), labels.squeeze())
        else:
          loss = loss_fct(logits, labels)
      elif self.config.problem_type == "single_label_classification":
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      elif self.config.problem_type == "multi_label_classification":
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

    if not return_dict:
      output = (logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class BertConcatCLS(BertPreTrainedModel):
  def __init__(self, config, n_features, dropout=0.3, hidden_layers_to_concat=1):
    super().__init__(config)
    print(config)
    self.num_labels = config.num_labels
    self.config = config
    self.n_features = n_features
    self.hidden_layers_to_concat = hidden_layers_to_concat
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear((config.hidden_size*self.hidden_layers_to_concat)+self.n_features, config.num_labels)
    self.post_init()

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    return_dict=None,
    features=None
  ):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=True,
      return_dict=return_dict,
    )

    # --------------------------------------------------------------------------
    # concat cls tokens
    cls_tokens = []
    for i in range(1, self.hidden_layers_to_concat + 1):
        cls_tokens.append(outputs.hidden_states[-1 * i][:, 0, :])
    pooled_output = torch.cat(cls_tokens, dim=1)
    # --------------------------------------------------------------------------

    pooled_output = self.dropout(pooled_output)
    if features is not None: # concat global features
      pooled_output = torch.cat((pooled_output, features), dim=1)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
      if self.config.problem_type is None:
        if self.num_labels == 1:
            self.config.problem_type = "regression"
        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            self.config.problem_type = "single_label_classification"
        else:
            self.config.problem_type = "multi_label_classification"

      if self.config.problem_type == "regression":
        loss_fct = nn.MSELoss()
        if self.num_labels == 1:
          loss = loss_fct(logits.squeeze(), labels.squeeze())
        else:
          loss = loss_fct(logits, labels)
      elif self.config.problem_type == "single_label_classification":
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      elif self.config.problem_type == "multi_label_classification":
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

    if not return_dict:
      output = (logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class BertAverageCLS(BertPreTrainedModel):
  def __init__(self, config, n_features, dropout=0.3, hidden_layers_to_average=1):
    super().__init__(config)
    print(config)
    self.num_labels = config.num_labels
    self.config = config
    self.n_features = n_features
    self.hidden_layers_to_average = hidden_layers_to_average
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(config.hidden_size+self.n_features, config.num_labels)
    self.post_init()

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    return_dict=None,
    features=None
  ):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=True,
      return_dict=return_dict,
    )

    # --------------------------------------------------------------------------
    # average cls tokens
    cls_tokens = []
    for i in range(1, self.hidden_layers_to_average + 1):
        cls_tokens.append(outputs.hidden_states[-1 * i][:, 0, :])
    pooled_output = torch.mean(torch.stack(cls_tokens), dim=0)
    # --------------------------------------------------------------------------

    pooled_output = self.dropout(pooled_output)
    if features is not None: # concat global features
      pooled_output = torch.cat((pooled_output, features), dim=1)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
      if self.config.problem_type is None:
        if self.num_labels == 1:
            self.config.problem_type = "regression"
        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            self.config.problem_type = "single_label_classification"
        else:
            self.config.problem_type = "multi_label_classification"

      if self.config.problem_type == "regression":
        loss_fct = nn.MSELoss()
        if self.num_labels == 1:
          loss = loss_fct(logits.squeeze(), labels.squeeze())
        else:
          loss = loss_fct(logits, labels)
      elif self.config.problem_type == "single_label_classification":
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      elif self.config.problem_type == "multi_label_classification":
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

    if not return_dict:
      output = (logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )