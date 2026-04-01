import torch
from tqdm import tqdm

from FeatureDiversityLoss import softmax_feature_maps, preserve_avg_func


def compute_localizing_variance(loader, model, device):
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=4)
    used_features = model.model.get_used_features()
    arange = torch.arange(used_features.shape[0], device=device)
    zero_features = torch.zeros((used_features.shape[0],), device=device)
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):

            input = input.to(device)
            _, feature_maps, _ = model(input, with_feature_maps=True,
                                       with_image=False,
                                       with_final_features=True)
            feature_maps = feature_maps[:, used_features]
            flat_maps = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)

            if i == 0:
                counter = torch.zeros((flat_maps.shape[1], flat_maps.shape[2]), device=device)
            max_args = torch.argmax(flat_maps, dim=2)
            adder = flat_maps.sum(dim=2) != 0
            zero_features += (1 - adder.type(torch.int)).sum(dim=0)
            for j in range(max_args.shape[0]):
                counter[arange, max_args[j]] += 1 * adder[j]

            # assert torch.sum(counter, dim=1) == len(arange)

    highest_frequency = torch.amax(counter, dim=1)
    highest_frequency = highest_frequency * len(loader.dataset) / (len(loader.dataset) - zero_features)
    return highest_frequency


def compute_average_softmax(loader, model, device, return_all_features=False):
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=4)
    used_features = model.model.get_used_features()
    if return_all_features:
        used_features = torch.arange(model.model.init_features, device=device)
    answer = torch.zeros(len(used_features), device=device)
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            _, feature_maps, features = model(input, with_feature_maps=True,
                                              with_image=False,
                                              with_final_features=True)

            if return_all_features and features.shape[1] != model.model.init_features:
                used_features = model.model.get_used_features()

            feature_maps = feature_maps[:, used_features]
            softmaxed = softmax_feature_maps(feature_maps)
            max_map = torch.amax(softmaxed, dim=(2, 3))
            answer += torch.sum(max_map, dim=0)
    average_loc = answer / len(loader.dataset)
    return average_loc.cpu().numpy()


def compute_average_softmax_att(features_train, loader, model, device, return_all_features=False):
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=4)
    used_features = model.model.get_used_features()
    model.model.return_att = True
    if return_all_features:
        used_features = torch.arange(model.model.init_features, device=device)
    weights = (features_train / features_train.sum(dim=1, keepdim=True)).to(device)
    answer = torch.zeros(len(used_features), device=device)
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            attention, features = model(input, with_feature_maps=True,
                                        with_image=False,
                                        with_final_features=True)

            if return_all_features and features.shape[1] != model.model.init_features:
                used_features = model.model.get_used_features()

            feature_maps = feature_maps[:, used_features]
            softmaxed = softmax_feature_maps(feature_maps)
            max_map = torch.amax(softmaxed, dim=(2, 3))
            answer += torch.sum(max_map * weights[i * input.shape[0]: (i + 1) * input.shape[0]], dim=0)
    average_loc = answer / len(loader.dataset)
    model.model.return_att = False
    return average_loc.cpu().numpy()


def compute_average_softmax_maxdef(features_train, loader, model, device, return_all_features=False, mean_scaled=False):
    bs = 16
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=bs, shuffle=False, num_workers=4)
    used_features = model.model.get_used_features()
    model.model.skip_prediction = True
    if return_all_features:
        used_features = torch.arange(model.model.init_features, device=device)
    weights = (features_train / features_train.sum(dim=1, keepdim=True)).to(device)
    answer = torch.zeros(len(used_features), device=device)
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            _, feature_maps, features = model(input, with_feature_maps=True,
                                              with_image=False,
                                              with_final_features=True)

            if return_all_features and features.shape[1] != model.model.init_features:
                used_features = model.model.get_used_features()

            feature_maps = feature_maps[:, used_features]
            if mean_scaled:
                feature_maps = feature_maps / torch.clamp(torch.mean(torch.abs(feature_maps),
                                                                     dim=(2, 3), keepdim=True), min=1e-5)
            softmaxed = softmax_feature_maps(feature_maps)
            max_map = torch.amax(softmaxed, dim=(2, 3))
            answer += torch.sum(max_map * weights[i * input.shape[0]: (i + 1) * input.shape[0]], dim=0)
    average_loc = answer / len(loader.dataset)
    model.model.skip_prediction = False
    return average_loc.cpu().numpy()


def compute_actual_localizing_variance(loader, model, device, base, exp, norm, return_all_features=False):
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=4)
    used_features = model.model.get_used_features()
    if return_all_features:
        used_features = torch.arange(model.model.init_features, device=device)
    maps = []
    all_features = []
    top_x = None
    if base is not None:
        top_x = int(base) * 10 ** int(exp)
        norm = softmax_feature_maps if norm is "S" else None
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            _, feature_maps, features = model(input, with_feature_maps=True,
                                              with_image=False,
                                              with_final_features=True)
            if return_all_features and features.shape[1] != model.model.init_features:
                used_features = model.model.get_used_features()
            feature_maps = feature_maps[:, used_features]
            flat_maps = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)
            all_features.append(features[:, used_features].to("cpu"))

            maps.append(flat_maps.to("cpu"))
    maps = torch.cat(maps, dim=0)
    if top_x is not None:
        all_features = torch.cat(all_features, dim=0)
        torch.testing.assert_allclose(maps.mean(dim=2), all_features)
        top_x = torch.topk(all_features, k=top_x, dim=0)
        relevant_maps = torch.gather(maps, index=top_x.indices[..., None].repeat(1, 1, maps.shape[2]), dim=0)
        torch.testing.assert_allclose(relevant_maps.mean(dim=2), top_x.values)
        normed_relevant_maps = norm(relevant_maps)
        std = torch.std(normed_relevant_maps, dim=0)
        average_feature_variance = torch.mean(std, dim=1)
    else:

        zero_maps = maps.sum(dim=(2)) == 0  #
        avgs = maps.mean(dim=2) / maps.mean(dim=(0, 2))

        zero_maps = zero_maps.unsqueeze(dim=2)
        # maps = softmax_feature_maps(maps)  #
        # averaged_softmaxd = torch.mean(maps, dim=0)
        maps = torch.sum(maps, dim=0)
        maps = softmax_feature_maps(maps[None])[0]
        top2 = torch.topk(maps, k=2, dim=1).values
        average_feature_variance = top2[:, 1]
    # top2 = torch.topk(maps, k=2, dim=2).values
    # average_feature_variance = torch.mean(top2[:, :, 1:] * avgs.unsqueeze(2), dim=0)[:, 0]

    # relevant_mean = (maps * ~zero_maps).sum(dim=0) / (maps.shape[0] - zero_maps.sum(dim=0))
    # relevant_squares = (maps - relevant_mean[None]) ** 2
    # combined_var = (relevant_squares * ~zero_maps * avgs.unsqueeze(2)).sum(dim=0) / (
    #         maps.shape[0] - zero_maps.sum(dim=0))
    # average_feature_variance = combined_var.median(dim=1)[0] # Definitely best so far
    # # variance = torch.var(maps, dim=0)
    # # average_feature_variance = torch.mean(variance, dim=1)
    # average_feature_variance = combined_var.std(dim=1)
    return average_feature_variance


def cog_batch(maps):
    # Input shape: ( batch xfeatures, height, width), softmaxed distribution
    # Output shape: ( batch xf features, 2)
    with torch.no_grad():
        answer = torch.zeros((maps.shape[0], maps.shape[1], 2), device=maps.device)
        assert torch.allclose(torch.sum(maps, dim=(2, 3)),
                              torch.ones((maps.shape[0], maps.shape[1],), device=maps.device))
        # maps = softmax_feature_maps(maps)
        x_summed = torch.sum(maps, dim=3)
        aranged = torch.arange(1, maps.shape[2] + 1, device=maps.device)
        x = torch.sum(x_summed * aranged[None, :], dim=2)
        answer[:, :, 0] += x
        y_summed = torch.sum(maps, dim=2)
        y = torch.sum(y_summed * aranged[None, :], dim=2)
        answer[:, :, 1] += y
        diff = answer - (aranged[-1] + aranged[0]) / 2
        norm_diff = torch.norm(diff, dim=2)
    return answer, diff, norm_diff


def cog(maps):
    # Input shape: ( features, height, width), softmaxed distribution
    # Output shape: ( features, 2)
    with torch.no_grad():
        answer = torch.zeros((maps.shape[0], 2), device=maps.device)
        assert torch.allclose(torch.sum(maps, dim=(2, 1)), torch.ones(maps.shape[0], device=maps.device))
        # maps = softmax_feature_maps(maps)
        x_summed = torch.sum(maps, dim=2)
        aranged = torch.arange(1, maps.shape[2] + 1, device=maps.device)
        x = torch.sum(x_summed * aranged[None, :], dim=1)
        answer[:, 0] += x
        y_summed = torch.sum(maps, dim=1)
        y = torch.sum(y_summed * aranged[None, :], dim=1)
        answer[:, 1] += y
        diff = answer - (aranged[-1] + aranged[0]) / 2
        norm_diff = torch.norm(diff, dim=1)
    return answer, diff, norm_diff.cpu()


def max_diff(maps):
    middle = maps.shape[2] / 2
    argmaxes = torch.argmax(maps.flatten(-2), dim=(2))
    x, y = torch.div(argmaxes, maps.shape[2], rounding_mode='trunc'), torch.remainder(argmaxes, maps.shape[2])
    diff = torch.stack((x, y), dim=2) - middle
    norm_diff = torch.norm(diff, dim=2)
    return norm_diff


def fast_diff_to_edge(maps):
    argmaxes = torch.argmax(maps.flatten(-2), dim=(2))
    x, y = torch.div(argmaxes, maps.shape[2], rounding_mode='trunc'), torch.remainder(argmaxes, maps.shape[2])
    # get lowest distance of x,y to any point with
    diffs = torch.zeros((maps.shape[:-1]), device=maps.device)
    max_tensor = torch.stack((x, y), dim=2)

    right_edge = torch.abs(x - (maps.shape[2] - 1))
    left_edge = x
    top_edge = y
    bottom_edge = torch.abs(y - (maps.shape[2] - 1))
    diffs = torch.stack((right_edge, left_edge, top_edge, bottom_edge), dim=2)
    diffs = torch.min(diffs, dim=2)[0]
    return diffs


def diff_to_edge(maps):
    argmaxes = torch.argmax(maps.flatten(-2), dim=(2))
    x, y = torch.div(argmaxes, maps.shape[2], rounding_mode='trunc'), torch.remainder(argmaxes, maps.shape[2])
    # get lowest distance of x,y to any point with
    diffs = torch.zeros((maps.shape[:-1]), device=maps.device)
    max_tensor = torch.stack((x, y), dim=2)
    for i in range(maps.shape[2]):
        this_diffs = torch.zeros((*maps.shape[:-2], 4), device=maps.device)
        this_diffs[..., 0] = torch.norm(
            max_tensor - torch.tensor((i, 0), dtype=max_tensor.dtype, device=max_tensor.device).type(torch.float),
            dim=2)
        this_diffs[..., 1] = torch.norm(
            max_tensor - torch.tensor((i, maps.shape[2] - 1), dtype=max_tensor.dtype, device=max_tensor.device).type(
                torch.float), dim=2)
        this_diffs[..., 2] = torch.norm(
            max_tensor - torch.tensor((0, i), dtype=max_tensor.dtype, device=max_tensor.device).type(torch.float),
            dim=2)
        this_diffs[..., 3] = torch.norm(
            max_tensor - torch.tensor((maps.shape[2] - 1, i,), dtype=max_tensor.dtype, device=max_tensor.device).type(
                torch.float), dim=2)
        diffs[:, :, i] = torch.amin(this_diffs, dim=2)

    norm_diff = torch.amin(diffs, dim=2)
    return norm_diff


def check_on_edge(maps):
    argmaxes = torch.argmax(maps.flatten(-2), dim=(2))
    x, y = torch.div(argmaxes, maps.shape[2], rounding_mode='trunc'), torch.remainder(argmaxes, maps.shape[2])
    # get lowest distance of x,y to any point with

    max_tensor = torch.stack((x, y), dim=2)
    is_max = max_tensor == maps.shape[2] - 1
    is_min = max_tensor == 0
    on_edge = torch.logical_or(is_max, is_min)
    on_edge = torch.any(on_edge, dim=2)
    return on_edge.float()


def max_diff_to_frac(maps, frac):
    argmaxes = torch.argmax(maps.flatten(-2), dim=(2))
    x, y = torch.div(argmaxes, maps.shape[2], rounding_mode='trunc'), torch.remainder(argmaxes, maps.shape[2])
    # get lowest distance of x,y to any point with

    max_tensor = torch.stack((x, y), dim=2)
    center_size = int(maps.shape[2] * frac)
    x_start = int((maps.shape[2] - center_size) / 2)
    y_start = int((maps.shape[2] - center_size) / 2)
    x_end = x_start + center_size - 1
    y_end = y_start + center_size - 1
    diffs = torch.zeros((*maps.shape[:-2], center_size), device=maps.device)
    for i in range(center_size):
        this_diffs = torch.zeros((*maps.shape[:-2], 4), device=maps.device)
        this_diffs[..., 0] = torch.norm(
            max_tensor - torch.tensor((x_start + i, y_start), dtype=max_tensor.dtype, device=max_tensor.device).type(
                torch.float),
            dim=2) * (y < y_start)
        this_diffs[..., 1] = torch.norm(
            max_tensor - torch.tensor((x_start + i, y_end), dtype=max_tensor.dtype, device=max_tensor.device).type(
                torch.float), dim=2) * (y > y_end)
        this_diffs[..., 2] = torch.norm(
            max_tensor - torch.tensor((x_start, y_start + i), dtype=max_tensor.dtype, device=max_tensor.device).type(
                torch.float),
            dim=2) * (x < x_start)
        this_diffs[..., 3] = torch.norm(
            max_tensor - torch.tensor((x_end, y_start + i,), dtype=max_tensor.dtype, device=max_tensor.device).type(
                torch.float), dim=2) * (x > x_end)
        this_diffs[this_diffs == 0] = torch.inf
        diffs[:, :, i] = torch.amin(this_diffs, dim=2)

    diffs[diffs[:, :] == torch.inf] = 0
    norm_diff = torch.amin(diffs, dim=2)
    return norm_diff


def compute_sciddle(loader, model, device, features, frac, return_all_features=False, ):
    bs = 24
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=bs, shuffle=False, num_workers=0)  # 4)
    used_features = model.model.get_used_features()
    model.model.skip_prediction = True
    if return_all_features:
        used_features = torch.arange(model.model.init_features, device=device)
    weights = (features / features.sum(dim=1, keepdim=True)).to(device)
    answer = torch.zeros(features.shape[1], device=device)
    no_scale = torch.zeros(features.shape[1], device=device)

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(loader), total=len(loader)):
            input = input.to(device)
            _, feature_maps, features = model(input, with_feature_maps=True,
                                              with_image=False,
                                              with_final_features=True)

            if return_all_features and features.shape[1] != model.model.init_features:
                used_features = model.model.get_used_features()

            feature_maps = feature_maps[:, used_features]
            if frac == 1:
                diff_to_middle = 1 / (diff_to_edge(feature_maps) + 1)
            elif frac == -1:
                diff_to_middle = check_on_edge(feature_maps)
            else:
                diff_to_middle = max_diff_to_frac(feature_maps, frac)
            no_scale += torch.sum(diff_to_middle, dim=0)
            answer += torch.sum(diff_to_middle * weights[i * bs:(i + 1) * bs], dim=0)
            #  flat_maps = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)
    model.model.skip_prediction = False
    return -answer.cpu().numpy()


def compute_dsmiddle(loader, model, device, features, degree, return_all_features=False, ):
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=4)
    used_features = model.model.get_used_features()
    if return_all_features:
        used_features = torch.arange(model.model.init_features, device=device)
    weights = (features / features.sum(dim=1, keepdim=True)).to(device)
    answer = torch.zeros(features.shape[1], device=device)
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            _, feature_maps, features = model(input, with_feature_maps=True,
                                              with_image=False,
                                              with_final_features=True)

            if return_all_features and features.shape[1] != model.model.init_features:
                used_features = model.model.get_used_features()

            feature_maps = feature_maps[:, used_features]
            diff_to_middle = max_diff(feature_maps) ** degree
            answer += torch.sum(diff_to_middle * weights[i * 64:(i + 1) * 64], dim=0)
            #  flat_maps = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)

    return answer.cpu().numpy()


def compute_smiddle(loader, model, device, features, return_all_features=False):
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=4)
    used_features = model.model.get_used_features()
    if return_all_features:
        used_features = torch.arange(model.model.init_features, device=device)
    weights = (features / features.sum(dim=1, keepdim=True)).to(device)
    answer = torch.zeros(features.shape[1], device=device)
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            _, feature_maps, features = model(input, with_feature_maps=True,
                                              with_image=False,
                                              with_final_features=True)

            if return_all_features and features.shape[1] != model.model.init_features:
                used_features = model.model.get_used_features()

            feature_maps = feature_maps[:, used_features]
            middle_ness, diff_middle_ness, norm_diff = cog_batch(softmax_feature_maps(feature_maps))
            answer += torch.sum(norm_diff * weights[i * 64:(i + 1) * 64], dim=0)
            #  flat_maps = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)

    return answer.cpu().numpy()


def compute_middle(loader, model, device, return_all_features=False):
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=4)
    used_features = model.model.get_used_features()
    if return_all_features:
        used_features = torch.arange(model.model.init_features, device=device)
    average_map = None
    all_features = []
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            _, feature_maps, features = model(input, with_feature_maps=True,
                                              with_image=False,
                                              with_final_features=True)

            if return_all_features and features.shape[1] != model.model.init_features:
                used_features = model.model.get_used_features()

            feature_maps = feature_maps[:, used_features]
            #  flat_maps = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)
            if average_map is None:
                average_map = torch.zeros(feature_maps.shape[1:], device=device)
            average_map += feature_maps.sum(dim=0)
            all_features.append(features[:, used_features].to("cpu"))
    real_average_map = softmax_feature_maps(average_map[None])
    middle_ness, diff_middle_ness, norm_diff = cog(real_average_map[0])

    return norm_diff


def compute_diversity(loader, model, device, scalebias, return_all_features=False):
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=4)
    used_features = model.model.get_used_features()
    if return_all_features:
        used_features = torch.arange(model.model.init_features, device=device)
    average_map = None
    all_features = []
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            _, feature_maps, features = model(input, with_feature_maps=True,
                                              with_image=False,
                                              with_final_features=True)

            if return_all_features and features.shape[1] != model.model.init_features:
                used_features = model.model.get_used_features()

            feature_maps = feature_maps[:, used_features]
            flat_maps = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)
            if average_map is None:
                average_map = torch.zeros(flat_maps.shape[1:], device=device)
            average_map += flat_maps.sum(dim=0)
            all_features.append(features[:, used_features].to("cpu"))
    real_average_map = softmax_feature_maps(average_map[None])
    feature_cat = torch.cat(all_features, dim=0)
    weights = (feature_cat / feature_cat.sum(dim=1, keepdim=True)).to(device)
    average_feature_variance = torch.zeros(feature_cat.shape[1], device=device)
    average_max = torch.zeros(feature_cat.shape[1], device=device)
    start = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            _, feature_maps, features = model(input, with_feature_maps=True,
                                              with_image=False,
                                              with_final_features=True)
            feature_maps = feature_maps[:, used_features]
            feature_maps = softmax_feature_maps(feature_maps)
            average_max += torch.sum(torch.amax(feature_maps, dim=(2, 3)) * weights[
                                                                            start:start + feature_maps.shape[0]], dim=0)
            flat_maps = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)
            average_feature_variance += ((real_average_map - flat_maps) ** 2 * weights[
                                                                               start:start + flat_maps.shape[
                                                                                   0]].unsqueeze(2)).sum(dim=(0, 2))
            start += flat_maps.shape[0]
    if len(scalebias) > 0:
        if scalebias == "sq":
            max_per_feature = torch.sqrt(average_max)
        elif scalebias == "s":
            max_per_feature = average_max
        elif scalebias == "ssq":
            max_per_feature = torch.sqrt(average_max)
            init_scale = average_feature_variance.max() / average_feature_variance.min()
            max_scale = max_per_feature.max() / max_per_feature.min()
            additional_scale = init_scale / max_scale
            max_per_feature *= additional_scale
        average_feature_variance *= max_per_feature

    return average_feature_variance.cpu().numpy()


if __name__ == '__main__':
    test_batched_maps = torch.rand((2, 7, 28, 28))
    # print(check_on_edge(test_batched_maps))
    # print(max_diff_to_frac(test_batched_maps, 0.5))
    print(diff_to_edge(test_batched_maps))
    print(fast_diff_to_edge(test_batched_maps))
    print("Diff ", diff_to_edge(test_batched_maps) - fast_diff_to_edge(test_batched_maps))
    print(max_diff(test_batched_maps))

    test_maps = torch.rand(7, 10, 10)
    print(cog(softmax_feature_maps(test_maps[None])[0]))
