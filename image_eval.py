from tqdm import trange, tqdm, tqdm_notebook
import os
import torch

def training(net, device, train_loader, optimizer, n_epoch, save_interval, loss_func, PATH = os.getcwd()):
    #setting network to train mode just incase there are modules which differntiate between train and eval
    net.train()
    total_loss = []
    for epoch in range(n_epoch):
        epoch_loss = []
        for mini_batch_idx, (image, label) in enumerate(train_loader):
            #select device for minibatch sample eithe CPU or GPU
            sample, g_truth = image.to(device), label.to(device)
            #clearing out grad thing
            optimizer.zero_grad()
            #getting pred 
            pred = net(sample)
            #calc the loss
            loss = loss_func(pred, g_truth )
            #get the gradient
            loss.backward()
            #apply the gradietn 
            optimizer.step()
            epoch_loss.append(loss.item())
            change_loss = (epoch_loss[0] - loss.item())/epoch_loss[0] * 100
            round(change_loss, 2)
            print("\r Loss: {}\t Epoch {} is {} % Complete. \t The Loss has decreased by: {} %".format(
                round(loss.item(),2), 
                epoch+1,
                round(mini_batch_idx/(len(train_loader.sampler)/train_loader.batch_size)*100,2),
                round(change_loss,2)), end = "")
          
        #print("\r The Loss has decreased by: {} %".format(change_loss), end = "")
            #use tqdm for total training progress
            #print("\r Loss: {}\t Epoch {} is {} % Complete.".format(loss.item(), epoch+1,
            #      (mini_batch_idx/(len(train_loader.sampler)/train_loader.batch_size))*100), end = "")
        total_loss.append(epoch_loss)
        if(epoch % save_interval == 0):
            print("\nThis is Epoch: {} of {} \t Training {} % Complete.".format(epoch+1, n_epoch, (epoch+1/n_epoch)*100))
            print("\nThe Loss is: {}".format(loss.item()))
            p = os.path.join(PATH, "model_chkpnt_epoch_{}_.tar".format(epoch))
            T.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'percent trained': (epoch/n_epoch)*100,
                'loss history': total_loss
            }, p)


#######

#######
def testing(net, device, test_loader, loss_func):
    test_loss = []
    net.eval()
    correct = 0
    with torch.no_grad():
        tqdm.write("\r Calculating the accuracy: ", end = "")
        for i,  (sample, g_truth) in enumerate(tqdm(test_loader)):
            image, label = sample.to(device), g_truth.to(device)
            pred_vec = net(image)
            loss = loss_func(pred_vec, label).item() #  summing up the loss
            pred = pred_vec.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()
            test_loss.append(loss)
    acc = correct/len(test_loader.sampler)*100
    tqdm.write("Accuracy: {} Correct {}\n".format(acc, correct), end = "")
    return test_loss

    

### if loading from checkpoint: ###
# checkpoint = torch.load(PATH)
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# net.eval()
# print(net)



