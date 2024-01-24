import matplotlib.pyplot as plt

log_raw = open("../logs/fo_prune_config.txt", "r")

datas = []
data_count = 0

# example line 1: Params 0.01% alive (2114/21289802) parameters
# example line 2: Epoch 1/50	lr: 0.010000 | train_acc: 12.36%  	train_loss: 2.2892  	val_acc: 10.96%  	val_loss: 2.2750

while True:
    line = log_raw.readline()
    if not line: break

    if line[:6] == "Params":
        pruning_percentage = float(line.split()[1][:-1])
        num_params = int(line.split()[3].split("/")[0][1:])
        datas.append({
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

        datas[data_count-1]["epoch"].append(epoch)
        datas[data_count-1]["lr"].append(lr)
        datas[data_count-1]["train_acc"].append(train_acc)
        datas[data_count-1]["train_loss"].append(train_loss)
        datas[data_count-1]["val_acc"].append(val_acc)
        datas[data_count-1]["val_loss"].append(val_loss)

for data in datas:
    data["best_val_acc"] = max(data["val_acc"])

log_raw.close()

for data in datas:
    plt.plot(data["epoch"], data["train_acc"], label=f"{data['pruning_percentage']}%")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.legend()
plt.savefig("../graphs/fo_prune_config_train_acc.png")
plt.clf()

for data in datas:
    plt.plot(data["epoch"], data["train_loss"], label=f"{data['pruning_percentage']}%")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig("../graphs/fo_prune_config_train_loss.png")
plt.clf()

for data in datas:
    plt.plot(data["epoch"], data["val_acc"], label=f"{data['pruning_percentage']}%")
plt.xlabel("Epoch") 
plt.ylabel("Validation Accuracy")
plt.legend()
plt.savefig("../graphs/fo_prune_config_val_acc.png")
plt.clf()

for data in datas:
    plt.plot(data["epoch"], data["val_loss"], label=f"{data['pruning_percentage']}%")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.savefig("../graphs/fo_prune_config_val_loss.png")
plt.clf()

# plot: x-axis is pruning percentage in log-scale, y-axis is best validation accuracy
# for each point of pruning percentage, write the number of parameters
pruning_percentages = [data["pruning_percentage"] for data in datas]
num_params = [data["num_params"] for data in datas]
best_val_accs = [data["best_val_acc"] for data in datas]

plt.plot(pruning_percentages, best_val_accs)
plt.scatter(pruning_percentages, best_val_accs)
for i in range(len(pruning_percentages)):
    plt.annotate(f"{num_params[i]}", (pruning_percentages[i], best_val_accs[i]), ha='right', rotation=-45)
plt.xlabel("Pruning Percentage")
plt.ylabel("Best Validation Accuracy")
plt.xscale("log")
plt.savefig("../graphs/fo_prune_config_best_val_acc.png")
plt.clf()