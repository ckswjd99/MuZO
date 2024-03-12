import matplotlib.pyplot as plt

log_mufo_raw = open("../logs/mutate_fo.txt", "r")

datas_mufo = []
data_count = 0

while True:
    line = log_mufo_raw.readline()
    if not line: break

    if line[:6] == "Params":
        pruning_percentage = float(line.split()[1][:-1]) / 100
        num_params = int(line.split()[3].split("/")[0][1:])
        datas_mufo.append({
            "pruning_percentage": pruning_percentage,
            "num_params": num_params,
            "epoch": [],
            "lr": [],
            "train_acc": [],
            "train_loss": [],
            "val_acc": [],
            "val_loss": []
        })
    elif line[:5] == "Epoch":
        epoch = int(line.split("/")[0].split()[1])
        lr = float(line.split()[3])
        train_acc = float(line.split()[6][:-1])
        train_loss = float(line.split()[8])
        val_acc = float(line.split()[10][:-1])
        val_loss = float(line.split()[12])

        datas_mufo[data_count-1]["epoch"].append(epoch)
        datas_mufo[data_count-1]["lr"].append(lr)
        datas_mufo[data_count-1]["train_acc"].append(train_acc)
        datas_mufo[data_count-1]["train_loss"].append(train_loss)
        datas_mufo[data_count-1]["val_acc"].append(val_acc)
        datas_mufo[data_count-1]["val_loss"].append(val_loss)

for data in datas_mufo:
    data["best_val_acc"] = max(data["val_acc"])

log_mufo_raw.close()


log_muconf_raw = open("../logs/fo_prune_config.txt", "r")

datas_muconf = []
data_count = 0

# example line 1: Params 0.01% alive (2114/21289802) parameters
# example line 2: Epoch 1/50	lr: 0.010000 | train_acc: 12.36%  	train_loss: 2.2892  	val_acc: 10.96%  	val_loss: 2.2750

while True:
    line = log_muconf_raw.readline()
    if not line: break

    if line[:6] == "Params":
        pruning_percentage = float(line.split()[1][:-1]) / 100
        num_params = int(line.split()[3].split("/")[0][1:])
        datas_muconf.append({
            "pruning_percentage": pruning_percentage,
            "num_params": num_params,
            "epoch": [],
            "lr": [],
            "train_acc": [],
            "train_loss": [],
            "val_acc": [],
            "val_loss": []
        })
    elif line[:5] == "Epoch":
        epoch = int(line.split("/")[0].split()[1])
        lr = float(line.split()[3])
        train_acc = float(line.split()[6][:-1])
        train_loss = float(line.split()[8])
        val_acc = float(line.split()[10][:-1])
        val_loss = float(line.split()[12])

        datas_muconf[data_count-1]["epoch"].append(epoch)
        datas_muconf[data_count-1]["lr"].append(lr)
        datas_muconf[data_count-1]["train_acc"].append(train_acc)
        datas_muconf[data_count-1]["train_loss"].append(train_loss)
        datas_muconf[data_count-1]["val_acc"].append(val_acc)
        datas_muconf[data_count-1]["val_loss"].append(val_loss)

for data in datas_muconf:
    data["best_val_acc"] = max(data["val_acc"])

log_muconf_raw.close()


# plot: x-axis is pruning percentage in log-scale, y-axis is best validation accuracy
# for each point of pruning percentage, write the number of parameters
pruning_percentages_muconf = [data["pruning_percentage"] for data in datas_muconf]
best_val_accs_muconf = [data["best_val_acc"] for data in datas_muconf]
pruning_percentage_mufo = [data["pruning_percentage"] for data in datas_mufo]
best_val_accs_mufo = [data["best_val_acc"] for data in datas_mufo]

plt.plot(pruning_percentages_muconf, best_val_accs_muconf, label="_nolegend_")
plt.scatter(pruning_percentages_muconf, best_val_accs_muconf)
plt.plot(pruning_percentage_mufo, best_val_accs_mufo, label="_nolegend_")
plt.scatter(pruning_percentage_mufo, best_val_accs_mufo)
plt.legend(["Pruning Configuration", "Mutation"])
plt.xlabel("Pruning Percentage")
plt.ylabel("Best Validation Accuracy")
plt.xscale("log")
plt.savefig("../graphs/fo_compare_muconf_val_acc.png")
plt.clf()

# plot: x-axis is pruning percentage in log-scale, y-axis is lowest validation loss
# for each point of pruning percentage, write the number of parameters
pruning_percentages_muconf = [data["pruning_percentage"] for data in datas_muconf]
best_val_losses_muconf = [min(data["val_loss"]) for data in datas_muconf]
pruning_percentage_mufo = [data["pruning_percentage"] for data in datas_mufo]
best_val_losses_mufo = [min(data["val_loss"]) for data in datas_mufo]

plt.plot(pruning_percentages_muconf, best_val_losses_muconf, label="_nolegend_")
plt.scatter(pruning_percentages_muconf, best_val_losses_muconf)
plt.plot(pruning_percentage_mufo, best_val_losses_mufo, label="_nolegend_")
plt.scatter(pruning_percentage_mufo, best_val_losses_mufo)
plt.legend(["Pruning Configuration", "Mutation"])
plt.xlabel("Pruning Percentage")
plt.ylabel("Lowest Validation Loss")
plt.xscale("log")
plt.savefig("../graphs/fo_compare_muconf_val_loss.png")
plt.clf()