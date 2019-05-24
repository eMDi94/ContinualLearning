import torch
import torch.optim as optim


def get_merged_and_counts(values):
    counts = torch.empty(len(values), dtype=torch.int)
    new_set = torch.empty(0)
    for idx, ex in enumerate(values):
        counts[idx] = len(ex)
        new_set = torch.cat((new_set, ex), dim=0)
    return new_set, counts


def get_labels(exemplar_sets, training_examples):
    new_set = [*exemplar_sets]
    new_set.extend(training_examples)
    labels = torch.empty(0, dtype=torch.long)
    for label, s in enumerate(new_set):
        current_labels = torch.tensor([label]).repeat(len(s))
        labels = torch.cat((labels, current_labels), dim=0)
    return labels


def iCaRL_update_representation(model, training_examples, exemplar_sets):
    """
    iCaRL function to update the representation of new classes
    :param model: The NN
    :param training_examples: Python Iterable which contains the new classes' samples
    :param exemplar_sets: Python Iterable which contain the stored examples
    :return: the updated model (for convenience only)
    """

    # Form combined training sets
    training_exs, training_counts = get_merged_and_counts(training_examples)
    exemplars, exemplars_counts = get_merged_and_counts(exemplar_sets)
    training_set = torch.cat((exemplars, training_exs), dim=0)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    with torch.enable_grad():
        model.train()

        optimizer.zero_grad()

        # After this add the new classes to the model
        model.add_classes(len(training_examples))

        # Run to obtain the scores for the new classes
        final_output = model(training_set)

        # Scores of the already known classes
        output = final_output[:, len(exemplar_sets)]

        # Scores of the new classes
        new_scores = final_output[:, -len(training_examples):]

        # Compute the loss function
        # First the distillation term with is simpler
        distillation_term = torch.mm(output, output.log()) + torch.mm((output - 1).neg(), (output - 1).neg().log())

        # The classification term is a little bit more difficult.
        labels = get_labels(exemplar_sets, training_examples).to(output.device)

        out = (new_scores - 1).neg().log()
        mask = torch.zeros_like(out, dtype=torch.long, device=out.device)
        mask[:, labels - len(exemplar_sets)] = 1
        out[mask] = new_scores[mask].log()
        classification_term = torch.sum(out, dim=1)

    # Compute the final loss
    loss = classification_term + distillation_term

    # Make the backward pass
    loss.backward()
    optimizer.step()

    return model
