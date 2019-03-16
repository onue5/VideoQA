def _parse_line(line):
    """ e.g., pred: (4, 40), gt: (4, 182)"""
    tokens = line.split("gt:")

    pred_str = tokens[0].strip()[7:-2]
    gt_str = tokens[1].strip()[1:-1]

    pred_tokens = pred_str.split(", ")
    gt_tokens = gt_str.split(", ")

    pred_start = int(pred_tokens[0])
    pred_end= int(pred_tokens[1])

    gt_start = int(gt_tokens[0])
    gt_end = int(gt_tokens[1])

    return pred_start, pred_end, gt_start, gt_end

tot_num = 0
n_pred_contain_gt = 0
n_gt_contain_pred = 0
overlap = 0

with open("testdata.txt") as fp:
    for line in fp:
        if line.startswith("pred:"):
            pred_start, pred_end, gt_start, gt_end = _parse_line(line)

            if pred_start <= gt_start <= pred_end and pred_start <= gt_end <= pred_end:
                n_pred_contain_gt += 1
            elif gt_start <= pred_start <= gt_end and gt_start <= pred_end <= gt_end:
                n_gt_contain_pred += 1
            elif pred_start <= gt_start <=pred_end or gt_start <= pred_start <=gt_end:
                overlap += 1

            tot_num += 1

print(float(n_pred_contain_gt)/ tot_num)
print(float(n_gt_contain_pred)/ tot_num)
print(float(overlap)/ tot_num)
