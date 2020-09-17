import torch


def smart_classify(
        resized, net, threshold=75, device='cuda',
        conf_threshold=0.9, debug=False):
    """
    First determined whether the cell is empty or not, checking the number of
    pixels different from zero.
    Then use a neural network to classify digits.
    """

    # Identify blank cells
    if (resized != resized.min()).sum() < threshold:
        return " "

    net.eval()
    iii = torch.Tensor(resized).to(device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out = net(iii)
        sm = torch.nn.functional.softmax(out, dim=1)

    if debug:
        print(f"Logits: {out}")
        print(f"Softmax: {sm}")

    _, P = torch.max(out, 1)

    digit = P.item()
    conf = sm.squeeze()[digit].item()

    if debug:
        print(f"Predicted digit: {digit} with confidence {conf:.3f}")

    if digit != 0 and conf > conf_threshold:
        return str(digit)
    else:
        return " "
