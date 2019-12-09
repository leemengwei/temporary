from __future__ import print_function
from IPython import embed
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import sys,os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import copy
import collections
import pandas as pd
import matplotlib.colors as colors
from matplotlib import cm
from torchviz import make_dot
import foschi_model
from random_force import get_random_force_seq
from mpl_toolkits.mplot3d import Axes3D

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_depth, output_size, device):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_depth = hidden_depth
        self.fc2 = nn.Linear(hidden_size, output_size)  
        self.relu = nn.ReLU()
        self.sp = nn.Softplus()
        self.th = nn.Tanh()
        self.sg = nn.Sigmoid()
        self.fcns = nn.ModuleList()   #collections.OrderedDict()
        self.bns = nn.ModuleList()   #collections.OrderedDict()
        for i in range(self.hidden_depth):
            self.fcns.append(nn.Linear(hidden_size, hidden_size).to(device))
            self.bns.append(nn.BatchNorm1d(hidden_size).to(device))
    def forward(self, x):
        #embed()
        out = self.fc1(x)
        for i in range(self.hidden_depth):
            out = self.fcns[i](out)
            out = self.bns[i](out)
            out = self.relu(out)
        out = self.fc2(out)
#        out = self.sg(out)
        return out

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    LOSS = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target) if args.lr<=1 else F.mse_loss(output, target)*args.lr
        loss.backward()
        LOSS += F.mse_loss(output, target, reduction='sum').item() # pile up all loss
        optimizer.step()
        if (batch_idx % args.log_interval == 0 or batch_idx == len(train_loader)-1)and(batch_idx!=0):
            pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
            pass
    train_loss_mean = LOSS/len(train_loader.dataset)
    print("Train Epoch: {} LOSS:{:.1f}, Average loss: {:.8f}".format(epoch, LOSS, train_loss_mean))
    return train_loss_mean

def validate(args, model, device, validate_loader):
    model.eval()
    LOSS = 0
    outputs_record = np.array([])
    targets_record = np.array([])
    with torch.no_grad():
        pbar = tqdm(validate_loader)
        for idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            LOSS += F.mse_loss(output, target, reduction='sum').item() # pile up all loss
            outputs_record = np.append(outputs_record, output.cpu())
            targets_record = np.append(targets_record, target.cpu())
            pbar.set_description('Validate: [{}/{} ({:.0f}%)]'.format(idx*len(data), len(validate_loader.dataset), 100.*idx/len(validate_loader)))
    validate_loss_mean = LOSS/len(validate_loader.dataset)
    print('Validate set LOSS: {:.1f}, Average loss: {:.8f}'.format(LOSS, validate_loss_mean))
    return validate_loss_mean, outputs_record, targets_record

def test(args, model, device, datas, Preprocessor, canadian_t, canadian_curve, wood_type, stress_level, color_list, middle_force):
    datas = datas.to(device)
    #model.train()
    model.eval()
    end_t = int(24*365*args.Test_years_to_run)
    process_alphas_list = np.empty(shape=(0, datas.shape[0]))
    t = 0
    t_list_for_alphas = np.array([])
    t_list_for_cum = np.array([])
    cum_list = np.array([])
    previous_alpha = Preprocessor.just_unormalize_alpha(datas[:,Preprocessor.which_is_alpha]).reshape(-1,1)
    if args.random_force:
        stress_level_norm = Preprocessor.just_normalize_force(stress_level)
    if args.mean != 0:
        datas[:, 0] = Preprocessor.just_normalize_wood_mean(args.mean)
        datas[:, 1] = Preprocessor.just_normalize_wood_cov(args.cov)
    with torch.no_grad():
        while t<end_t:
            if args.random_force:
                datas[:, Preprocessor.which_is_force] = stress_level_norm[int(t/24)]
            output = model(datas)
            t += args.Time_step
            damage = Preprocessor.just_unormalize_output(output)
            previous_alpha = torch.FloatTensor(np.clip(damage.cpu(), 0, 1)).to(device) + previous_alpha
            datas[:,Preprocessor.which_is_alpha] = Preprocessor.just_normalize_alpha(previous_alpha.cpu()).view(-1)
            idx = np.where(previous_alpha.cpu()>=1)[0]
            cum_list = np.append(cum_list, len(idx)/datas.shape[0])
            t_list_for_cum = np.append(t_list_for_cum, t)
            tmp_len = min(canadian_curve.shape[0], cum_list.shape[0])
            RR = np.round(np.corrcoef(canadian_curve[:tmp_len], cum_list[:tmp_len])[0,1]**2*100, 2)
            if (t%(24*365)==0) and args.Visualization:
                plt.clf()
                #Fig1:
                ax1 = plt.subplot(121)
                ax1.set_xlabel("Logged duration of Load (log hour)")
                ax1.set_ylabel("Internal Damage of Individuals (%)")
                if not args.Middle_process:
                    plt.text(0.1, 0.1, "Long-Term\nprediction\n\n, '--Middle_process' to show")
                else:
                    t_list_for_alphas = np.append(t_list_for_alphas, t)
                    process_alphas_list = np.vstack((process_alphas_list, previous_alpha.reshape(-1).cpu().numpy()))
                    ax1.plot(np.log10(t_list_for_alphas), np.clip(process_alphas_list,0,1)[:,::int(process_alphas_list.shape[1]/1000)])
                #Fig2:
                ax2 = plt.subplot(122)
                ax2.plot(np.log10(canadian_t), canadian_curve, '--', label="Canadian Model of %s %s"%(wood_type, np.round(stress_level, 2)), color=color_list[wood_type+str(stress_level)])
                ax2.plot(np.log10(t_list_for_cum), cum_list, label="Network Result of %s %s, D:%s%%"%(wood_type, np.round(stress_level, 2), "Constant array" if np.isnan(RR) else RR), color=color_list[wood_type+str(stress_level)])
                ax2.legend(loc="upper left", fontsize=8)
                ax2.set_xlabel("Logged duration of Load (log hour)")
                ax2.set_ylabel("Failure percentage (%)")
                plt.draw()
                plt.pause(0.001)
                print("Forward time: %s days, %s years, RR:%s%%"%(t/24, t/24/365, "Constant array" if np.isnan(RR) else RR), "Load:", stress_level[int(t/24)] if args.random_force else stress_level)
            if args.Alter_force_scaler!=1 and t==24*365*1:   #We alter force one year later
                physical_data = Preprocessor.unormalize_all(np.hstack((datas.cpu().numpy(),np.tile(0,(datas.shape[0],1)))))
                if args.Alter_force_scaler == 0:
                    print("Altering force to middle force of: %s"%middle_force)
                    physical_data[:, Preprocessor.which_is_force] = middle_force
                else:
                    print("Altering force to %s times"%args.Alter_force_scaler)
                    physical_data[:, Preprocessor.which_is_force] = physical_data[:, Preprocessor.which_is_force]*args.Alter_force_scaler
                datas = torch.FloatTensor(Preprocessor.normalize_all(physical_data)[:,:-1]).to(device)
                pass
    print("Forward time: %s days, %s years, RR:%s%%"%(t/24, t/24/365, "Constant array" if np.isnan(RR) else RR))
    plt.close()
    return t_list_for_cum, cum_list, t_list_for_alphas, process_alphas_list

class data_preprocessing(object):
    def __init__(self, _data_):
        data = copy.deepcopy(_data_)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.which_is_wood_mean = 0
        self.which_is_wood_cov = 1
        self.which_is_force = 2
        #self.which_is_alpha = int(np.array(list(set(list(np.where(data.max(axis=0)<1)[0])) & set(list(np.where(data.min(axis=0)>0)[0])))).min()) 
        self.which_is_alpha = 3
    def clean(self, _data_):
        data = copy.deepcopy(_data_)
        data = np.delete(data, np.where(data[:,self.which_is_alpha]==0), axis=0) #直接删除alpha等于零的所有点，因为要做log10......应该没有了， 都是1e-10
        data[np.where(data[:,-1]==0)]=np.hstack((data[np.where(data[:,-1]==0)][:,:-1], np.tile(data[np.where(data[:,-1]!=0)][:,-1].min(),(data[np.where(data[:,-1]==0)].shape[0],1)))) #对于损伤等于零的点，赋值样本中的最小损伤，之后方便取log10. (赋值方式只能这么麻烦。)
        return data
    def get_stastics(self, _data_):
        data = copy.deepcopy(_data_)
        data[:,self.which_is_alpha] = np.log10(data[:,self.which_is_alpha])  #Subtle variable damage
        data[:,-1] = np.log10(data[:,-1])  #Subtle variable damage
        self.mean = data.mean(axis=0)
        #self.mean = np.array([47.83/2, 0.41/2, 24.86194164, -5.95217765, 45.41826586, -8.64803982])
        self.mean = np.array([37.48443957, 0.3268635, 23.97584309, -5.33723825, 1.46837475, -7.18191736])
        self.std = data.std(axis=0)
        #self.std = np.array([47.83, 0.41,  5.07882468, 3.00969408, 8.8454502 , 3.4259231 ])
        self.std = np.array([10.38504219, 0.08763241, 6.60742912, 2.87682221, 0.98996641, 3.24795487])
        #print(self.mean, self.std)
    def normalize_all(self, _data_):
        data = copy.deepcopy(_data_)
        data = self.clean(data)
        data[:,self.which_is_alpha] = np.log10(data[:,self.which_is_alpha])  #Subtle variable alpha
        data[:,-1] = np.log10(data[:,-1])  #Subtle variable damage
        data = (data-self.mean)/self.std
        if np.isnan(data).any():
            print("Notice: There is NAN when normalizing input data...")
            print("This is either because that data series are all the same (All constant force? All no damage 0?) or there're just too few data (only one data)")
            embed()
            sys.exit()
        return data
    def just_normalize_alpha(self, _data_):
        data = copy.deepcopy(_data_)
        data = np.log10(data)
        data = (data-self.mean[self.which_is_alpha])/self.std[self.which_is_alpha]
        return data
    def just_normalize_output(self, _data_):
        data = copy.deepcopy(_data_)
        data = np.log10(data)
        data = (data-self.mean[-1])/self.std[-1]
        return data
    def just_normalize_force(self, _data_):
        data = copy.deepcopy(_data_)
        data = (data-self.mean[self.which_is_force])/self.std[self.which_is_force]
        return data
    def just_normalize_wood_mean(self, _data_):
        data = copy.deepcopy(_data_)
        data = (data-self.mean[self.which_is_wood_mean])/self.std[self.which_is_wood_mean]
        return data
    def just_normalize_wood_cov(self, _data_):
        data = copy.deepcopy(_data_)
        data = (data-self.mean[self.which_is_wood_cov])/self.std[self.which_is_wood_cov]
        return data
        
    def unormalize_all(self, _data_):
        data = copy.deepcopy(_data_)
        data = data*self.std+self.mean
        data[:,self.which_is_alpha] = 10**(data[:,self.which_is_alpha])  #Subtle variable alpha
        data[:,-1] = 10**(data[:,-1])  #Subtle variable damage
        return data
    def just_unormalize_alpha(self, _data_):
        data = copy.deepcopy(_data_)
        data = data*self.std[self.which_is_alpha] + self.mean[self.which_is_alpha]
        data = 10**(data)
        return data
    def just_unormalize_output(self, _data_):
        data = copy.deepcopy(_data_)
        data = data*self.std[-1] + self.mean[-1]
        data = 10**(data)
        return data

def get_color_list(wood_types, args):
        number_of_lines= len(wood_types)*args.num_of_forces
        cm_subsection = np.linspace(0.0, 1.0, number_of_lines) 
        color_values = [ cm.jet(x) for x in cm_subsection ]
        color_list = collections.OrderedDict()
        tmp_i = 0
        for wood_type in wood_types:
            for stress_level in wood_types[wood_type]:
                color_list[wood_type+str(stress_level)] =color_values[tmp_i]
                tmp_i+=1
        return color_list

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Wood_Damage Example')
    parser.add_argument('--num_of_batch', type=int, default=50,
                        help='input batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many mini-batches to wait before logging training status')
    parser.add_argument('--Visualization', action='store_true',\
            help="to show visualization, defualt false", default=False)
    parser.add_argument('--Middle_process', action='store_true',\
            help="if plot middle damage process, defaul false", default=False)
    parser.add_argument('--Restart',  action='store_true',\
            help='if restart', default = False)
    parser.add_argument('--Debug', action='store_true',\
            help='debug mode show prints, default clean mode', default = False)
    parser.add_argument('--Time_step', type=int,\
            help="time step when run canadian model, default 24", default = 24)
    parser.add_argument('--Time_interval', type=float,\
            help="seperate '1 Hour' to 'this' steps, default 1", default = 1)
    parser.add_argument('--num_of_forces', type=int,\
            help="Number of forces, default 2", default = 2)
    parser.add_argument('--Number_of_woods', type=int,\
            help="Number of woods, default 100", default = 100)
    parser.add_argument('--Years_to_run', type=float,\
            help="Years we run differential model", default = 5)
    parser.add_argument('--Test_years_to_run', type=float,\
            help="Years we run prediction NN model", default = 0.02)
    parser.add_argument('--workers', type=int,\
            help="Number of workers, default 10", default = 10)
    parser.add_argument('--Quick_data', action='store_true',\
            help='If Quick data from file, default False', default = False)
    parser.add_argument('--Save_model', action='store_true',\
            help='If Save model, default True', default = True)
    parser.add_argument('--Cuda_number', type=int,\
            help="Number of woods, default 0", default = 0)
    parser.add_argument('--RM', type=str,\
            help="Restart Model, default None, will use name current_best", default = None)
    parser.add_argument('--test_ratio', type=float,\
            help="Number of woods, default 0.3", default = 0.3)
    parser.add_argument('--wood_types', type=str,\
            help="Wood types, one of [All, Hemlock, SPF_Q1, SPF_Q2], default Hemlock", default = "Hemlock")
    parser.add_argument('--Alter_force_scaler', type=float,\
            help='If alter force when 1 year, default scaler 1.0, no alter, if its set to 0.0 then alter to middle force', default = 1.0)
    parser.add_argument('--mean', type=float,\
            help='wood mean for new type, default as hemlock:47.83', default = 0.0)
    parser.add_argument('--cov', type=float,\
            help='wood cov for new type, default as hemlock:0.41', default = 0.0)
    parser.add_argument('--random_force', action='store_true',\
            help='If use random force, default False', default = False)
    parser.add_argument('--random_force_interval', type=int,\
            help='depth of net', default = 7)
    parser.add_argument('--hidden_depth', type=int,\
            help='depth of net', default = 5)
    parser.add_argument('--hidden_width_scaler', type=int,\
            help='width scaler of net', default = 100)
    args = parser.parse_args()
    if args.epochs<=0:   #If epoch <0, it's Test mode.
        args.Restart=True
        args.Quick_data=True
    if args.random_force:
        assert args.Alter_force_scaler == 1.0, "You're using random force to sample, set Atler scaler to 1.0"
        if args.epochs < 0:
            assert args.RM is not None, "Strongly warning, your're running in random force mode, under this mode 'args.num_of_force' will refer to 'num_of_random_force' not the forces upon which the model is trained, thus set args.RM explictly for sure!"
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    model_best_path = "../output/current_best_%s_%s_%s.pth"%(args.num_of_batch, args.wood_types, args.num_of_forces) if args.RM is None  else args.RM
    torch.manual_seed(44)
    if use_cuda:
        device = torch.device("cuda", args.Cuda_number) 
    else:
        device = torch.device("cpu")
    global wood_types
    if args.wood_types == "All":
        wood_types = collections.OrderedDict([
			("Hemlock", list(np.linspace(5, 45, args.num_of_forces))), 
			("SPF_Q1", list(np.linspace(5, 45, args.num_of_forces))), 
			("SPF_Q2", list(np.linspace(5, 45, args.num_of_forces))),
			])
    elif args.wood_types == "Hemlock_and_SPF_Q1":
        wood_types = collections.OrderedDict([
			("Hemlock", list(np.linspace(5, 45, args.num_of_forces))), 
			("SPF_Q1", list(np.linspace(5, 45, args.num_of_forces))), 
			])
    elif args.wood_types == "Hemlock_and_SPF_Q2":
        wood_types = collections.OrderedDict([
			("Hemlock", list(np.linspace(5, 45, args.num_of_forces))), 
			("SPF_Q2", list(np.linspace(5, 45, args.num_of_forces))),
			])
    elif args.wood_types == "SPFS":
        wood_types = collections.OrderedDict([
			("SPF_Q1", list(np.linspace(5, 45, args.num_of_forces))), 
			("SPF_Q2", list(np.linspace(5, 45, args.num_of_forces))),
			])
    elif args.wood_types == "Hemlock":
       wood_types = collections.OrderedDict([
			("Hemlock", list(np.linspace(5, 45, args.num_of_forces))), 
			])
    elif args.wood_types == "SPF_Q1":
       wood_types = collections.OrderedDict([
			("SPF_Q1", list(np.linspace(5, 45, args.num_of_forces))), 
			])
    elif args.wood_types == "SPF_Q2":
       wood_types = collections.OrderedDict([
			("SPF_Q2", list(np.linspace(5, 45, args.num_of_forces))), 
			])
    elif args.wood_types == "Others":
       wood_types = collections.OrderedDict([
			("Others", list(np.linspace(5, 45, args.num_of_forces))),
			])
    else:
        print("Unkown wood type")
        sys.exit()
    '''
    if args.random_force:
        np.random.seed(55)
        random_force_seq = get_random_force_seq(args.num_of_forces, args.random_force_interval)
        for wood_type in wood_types.keys():   #Reset forces to random force seq.
            wood_types[wood_type] = random_force_seq
    if args.random_force and args.Visualization:
        plt.plot(random_force_seq[:,:1000].T)
        plt.title("%s Random load given"%args.num_of_forces)
        plt.xlabel("Duration of random load of first 1000 (hour)")
        plt.ylabel("Load series given (Mpa)")
        plt.show()
    '''
    global wood_type_coding
    wood_type_coding={}
    wood_type_coding["Hemlock"] = [47.83, 0.41]
    wood_type_coding["SPF_Q1"] = [48.90, 0.20]
    wood_type_coding["SPF_Q2"] = [25.77, 0.28]
    wood_type_coding["Others"] = [args.mean, args.cov]

    #Get data:
    Quick_name = "Quick_pickle_%s_%s_%s_forces_%s_years_%s%s%s.pkl"%(args.Number_of_woods, args.wood_types, args.num_of_forces, args.Years_to_run, args.Alter_force_scaler, "_random_force" if args.random_force else '', "_interval_%s"%args.random_force_interval if args.random_force else '')
    if args.Quick_data:
        objs = []
        print("Reading raw_data from pickle...%s"%Quick_name)
        with open("%s"%Quick_name, "rb") as pfile:
            while 1:
                try:
                    objs.append(pickle.load(pfile))
                except EOFError:
                    break
            if args.random_force:
                raw_data, t_of_canadian_curves, canadian_curves, random_force_seq = objs
                for wood_type in wood_types.keys():   #Reset forces to random force seq.
                    wood_types[wood_type] = random_force_seq
                if args.Visualization:
                    plt.plot(random_force_seq[:,:1000].T)
                    plt.title("%s Random load given"%args.num_of_forces)
                    plt.xlabel("Duration of random load of first 1000 (hour)")
                    plt.ylabel("Load series given (Mpa)")
                    plt.show()
            else:
                raw_data, t_of_canadian_curves, canadian_curves = objs
    else:
        raw_data, t_of_canadian_curves, canadian_curves = foschi_model.main_for_load_duration(args, wood_types, args.Time_step, wood_type_coding)
        #Remove same raw_data to balance training:
        #Unique均衡移动至数据生成程序。
        #print("Unique balancing...")
        #_, idx = np.unique(raw_data, axis=0, return_index=True)
        #idx.sort()
        #raw_data_part1 = raw_data[idx]
        #print("Compensating...")
	#Compensating暂缓。
        #idx_those_removed = np.array(list(set(list(range(raw_data.shape[0])))-set(idx)))
        #np.random.shuffle(idx_those_removed)
        #raw_data_part2 = raw_data[idx_those_removed[:int(raw_data_part1.shape[0]*0.025)]]
        #raw_data_part2 = raw_data[idx_those_removed[:int(raw_data_part1.shape[0]*0)]]
        #raw_data = np.vstack((raw_data_part1, raw_data_part2))
        print("Dumping raw_data pickle...",)
        pfile = open("%s"%Quick_name, "wb")
        pickle.dump(raw_data, pfile)
        pfile.close()
        pfile = open("%s"%Quick_name, "ab")
        pickle.dump(t_of_canadian_curves, pfile)
        pickle.dump(canadian_curves, pfile)
        pfile.close()
        if args.random_force:
            pfile = open("%s"%Quick_name, "ab")
            pickle.dump(random_force_seq, pfile)
            pfile.close()
            if args.Visualization:
                plt.plot(random_force_seq[:,:1000].T)
                plt.title("%s Random load given"%args.num_of_forces)
                plt.xlabel("Duration of random load of first 1000 (hour)")
                plt.ylabel("Load series given (Mpa)")
                plt.show()
    color_list = get_color_list(wood_types, args)
    embed()

    Preprocessor = data_preprocessing(raw_data)
    data = Preprocessor.clean(raw_data)
    Preprocessor.get_stastics(data)
    data= Preprocessor.normalize_all(data)
    inputs = torch.FloatTensor(data[:,:-1])
    targets = torch.FloatTensor(data[:,-1].reshape(-1, 1))
    whole_dataset = Data.TensorDataset(inputs, targets)
    train_dataset, validate_dataset = Data.random_split(whole_dataset, (len(whole_dataset)-int(len(whole_dataset)*args.test_ratio),int(len(whole_dataset)*args.test_ratio)))
    train_loader = Data.DataLoader( 
            dataset=train_dataset, 
            batch_size=int(len(train_dataset)/args.num_of_batch) if int(len(train_dataset)/args.num_of_batch)!=0 else len(train_dataset),
            shuffle=True,
            drop_last=True,
	        num_workers=args.workers,
            pin_memory=True
            )
    validate_loader = Data.DataLoader( 
            dataset=validate_dataset, 
            batch_size=int(len(validate_dataset)/args.num_of_batch) if int(len(validate_dataset)/args.num_of_batch)!=0 else len(validate_dataset),
            shuffle=True,
            drop_last=True,
	        num_workers=args.workers,
            pin_memory=True
            )
    #Input distribution:
    pd_data = pd.DataFrame(data)
    pd_data.columns = ['Mean of wood strength as input', "Cov of wood strength as input", "Load of wood strength as input", "Internal damage Alpha as input", " Strength related factor as input", "Step damage rate as output"]
    if args.Visualization and not args.random_force:
        #Failure curve:
        for wood_type in wood_types.keys():
            for stress_level in wood_types[wood_type]:
                plt.plot(np.log10(t_of_canadian_curves[wood_type+str(stress_level)]), canadian_curves[wood_type+str(stress_level)], color=color_list[wood_type+str(stress_level)])
        plt.title("%s, Altering force %s"%(args.wood_types, args.Alter_force_scaler))
        plt.xlabel("Logged duration of Load (log hour)")
        plt.ylabel("Failure percentage (%)")
        plt.show()
        #Response surface:
        ax = plt.subplot(projection='3d')
        for idx, wood_idx in enumerate(set(raw_data[:,0])):
            alpha, load, damage = raw_data[np.where(raw_data[:,0]==wood_idx), Preprocessor.which_is_alpha], raw_data[np.where(raw_data[:,0]==wood_idx), Preprocessor.which_is_force], raw_data[np.where(raw_data[:,0]==wood_idx), -1]
            alpha, load, damage = alpha.ravel(), load.ravel(), damage.ravel()
            ax.scatter(alpha[::100], load[::100], damage[::100], s=0.5, label="Wood_type:%s"%list(wood_types.keys())[idx])
        ax.set_xlabel("Internal State Alpha (Dimensionless)")
        ax.set_ylabel("Load (Dimensionless)")
        ax.set_zlabel("Internal Damage (Dimensionless)")
        ax.set_title("Response Surface Learned for neural network")
        plt.legend()
        plt.show()
        fig = plt.figure()
        axarr = []
        for _, i in enumerate(pd_data.columns):
            ax = fig.add_subplot(3, 2 , _+1)
            ax.hist(pd_data[i], bins=100)
            ax.set_title(i)
            ax.set_ylabel("Frequences")
            ax.grid(True)
            axarr.append(ax)
        plt.show()
    #embed()
    if not args.Quick_data:
        print("Data Generation Done...")
        sys.exit()

    #Load Model:
    input_size = inputs.shape[1] 
    hidden_size = inputs.shape[1]*args.hidden_width_scaler
    hidden_depth = args.hidden_depth
    output_size = 1
    model = NeuralNet(input_size, hidden_size, hidden_depth, output_size, device).to(device)
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
    #Show flow graph:
    #y = model(inputs.cuda()[:10,:])
    #g = make_dot(y.mean(), params=dict(model.named_parameters()))
    #g.view()

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    if args.Restart:
        model_restart_path = model_best_path if args.RM is None  else args.RM
        checkpoint = torch.load(model_restart_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss_history = checkpoint['train_loss_history']
        validate_loss_history = checkpoint['validate_loss_history']
        if args.Visualization:
            print("Restarting Model paramters loaded: %s"%model_restart_path)
            plt.plot(np.log10(validate_loss_history), linewidth=2, label="validate loss")
            plt.plot(np.log10(train_loss_history), linewidth=2, label="train loss")
            plt.title("Training/Validating loss over epoch")
            plt.legend()
            plt.xlabel("Training epoches")
            plt.ylabel("Training/Valdation loss")
            plt.show()
        else:
            pass
    else:
        epoch = 1
        train_loss_history = []
        validate_loss_history = []
    #Train and Validate:
    for epoch in range(epoch, args.epochs + 1):
        if args.Alter_force_scaler!=1.0:
            print("Warning, you're tring to do training, should be all constant force... Reset force altering scaler to 1.0")
            print("Sure to continue?")
            z=input()
        #print(model.bns[0].weight[-1].item(),model.fc1.weight[-1])
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        validate_loss, validate_outputs, validate_targets = validate(args, model, device, validate_loader)
        train_loss_history.append(train_loss)
        validate_loss_history.append(validate_loss)
        train_loss_history[0] = validate_loss_history[0]
        #plots:
        if args.Visualization:
            xaxis_train = range(len(train_loss_history))
            xaxis_validate = range(len(validate_loss_history))
            plt.clf()
            ax1 = plt.subplot(131)
            scope = 50
            ax1.scatter(xaxis_train[-scope:], 100*np.array(train_loss_history[-scope:]), color='k', s=0.85)
            ax1.scatter(xaxis_validate[-scope:], 100*np.array(validate_loss_history[-scope:]), color='blue', s=0.85)
            ax1.plot(xaxis_train[-scope:], 100*np.array(train_loss_history[-scope:]), color='k', label='trainloss')
            ax1.plot(xaxis_validate[-scope:], 100*np.array(validate_loss_history[-scope:]), color='blue', label="validateloss")
            ax1.set_xlabel("Epoches")
            ax1.set_ylabel("Loss")
            ax1.legend()
    
            ax2 = plt.subplot(132)
            idx = np.argsort(validate_targets)
            skipper = int(len(validate_targets)/1000) if int(len(validate_targets)/1000)!=0 else 1
            ax2.scatter(range(len(validate_targets))[::skipper], validate_targets[idx][::skipper], color='k', label="validate_targets", s=0.15)
            ax2.scatter(range(len(validate_outputs))[::skipper], validate_outputs[idx][::skipper], color='blue', label='validate_outputs', s=0.15)
            ax2.set_xlabel("Sorted damage label")
            ax2.set_ylabel("Training label verses output of processed data")
            ax2.legend()
    
            ax3 = plt.subplot(133)
            ax3.scatter(range(len(validate_targets))[::skipper], Preprocessor.just_unormalize_output(validate_targets[idx])[::skipper], color='k', label="damage_targets", s=0.15)
            ax3.scatter(range(len(validate_outputs))[::skipper], Preprocessor.just_unormalize_output(validate_outputs[idx])[::skipper], color='blue', label='damage_outputs', s=0.15)
            ax3.set_xlabel("Sorted damage")
            ax3.set_ylabel("Training label verses output of restored data")
            ax3.legend()
            plt.draw()
            plt.pause(0.001)
        if epoch <= 1:
            continue
        #Saving stuff:
        ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': train_loss_history,
                'validate_loss_history': validate_loss_history,
                }
        if args.Save_model:
            modeldir = "../output/epoch_%s_%s.pth"%(epoch, args.wood_types)
            print("Saving model at", modeldir)
            torch.save(ckpt, modeldir)
        if epoch>1:
            if validate_loss_history[-1] < np.array(validate_loss_history[:-1]).min():
                print("Saving current best model, epoch: %s"%epoch)
                torch.save(ckpt, model_best_path)
            else:
                print("Not best: %s>%s, Keep running..."%(validate_loss_history[-1], np.array(validate_loss_history[:-1]).min()))
        else:
            pass   #Pass epoch 1

    #Test:
    t_of_cum_lists, cum_lists, t_lists_for_alphas, real_alphas_lists = {}, {}, {}, {}

    for wood_type in list(wood_types.keys())[::-1]:
        for test_force in wood_types[wood_type]:
            print("Testing on: %s of %spsi"%(wood_type, test_force))
            print("Testing time length: %s years, number of woods %s"%(args.Test_years_to_run, args.Number_of_woods))
            test_data = foschi_model.for_generalization_starting_point(args, wood_type, test_force if not args.random_force else test_force[0], 1e-10, wood_type_coding)#1e-4)   #Leave pretreat for data_preprocessor
            test_data = np.hstack((test_data, np.tile(999,(test_data.shape[0],1))))
            test_input_data = torch.FloatTensor(Preprocessor.normalize_all(test_data))[:,:-1]
            middle_force = (wood_types[wood_type][0]+wood_types[wood_type][-1])*0.5
            #embed()
            t_of_cum_lists[wood_type+str(test_force)], cum_lists[wood_type+str(test_force)], t_lists_for_alphas, real_alphas_lists[wood_type+str(test_force)] = test(args, model, device, test_input_data, Preprocessor, t_of_canadian_curves[wood_type+str(test_force)], canadian_curves[wood_type+str(test_force)], wood_type, test_force, color_list, middle_force)

    #embed()
    RRs = np.array([])
    savings = pd.DataFrame([])
    plt.figure()
    for wood_type in wood_types:
        for stress_level in wood_types[wood_type]:
            #cum_lists[wood_type+str(stress_level)] = cum_lists[wood_type+str(stress_level)][1:]
            #t_of_cum_lists[wood_type+str(stress_level)] = t_of_cum_lists[wood_type+str(stress_level)][1:]
            tmp_len = min(canadian_curves[wood_type+str(stress_level)].shape[0], cum_lists[wood_type+str(stress_level)].shape[0])
            canadian_curves[wood_type+str(stress_level)] = canadian_curves[wood_type+str(stress_level)][:tmp_len]
            cum_lists[wood_type+str(stress_level)] = cum_lists[wood_type+str(stress_level)][:tmp_len]
            t_of_canadian_curves[wood_type+str(stress_level)] = t_of_canadian_curves[wood_type+str(stress_level)][:tmp_len]
            t_of_cum_lists[wood_type+str(stress_level)] = t_of_cum_lists[wood_type+str(stress_level)][:tmp_len]
            RR = np.round(np.corrcoef(canadian_curves[wood_type+str(stress_level)], cum_lists[wood_type+str(stress_level)])[0,1]**2*100, 2)
            plt.plot(np.log10(t_of_canadian_curves[wood_type+str(stress_level)]), canadian_curves[wood_type+str(stress_level)], '--', label="Canadian Model of %s %s"%(wood_type, np.round(stress_level, 2)), color=color_list[wood_type+str(stress_level)])
            plt.plot(np.log10(t_of_cum_lists[wood_type+str(stress_level)]), cum_lists[wood_type+str(stress_level)], label="Network Result of %s %s, D:%s%%"%(wood_type, np.round(stress_level, 2), "Constant array" if np.isnan(RR) else RR), color=color_list[wood_type+str(stress_level)])
            RRs = np.append(RRs, RR)
            savings["canadian_t_of_"+wood_type+str(stress_level)] = pd.Series(np.log10(t_of_canadian_curves[wood_type+str(stress_level)]))
            savings["canadian_curve_of_"+wood_type+str(stress_level)] = pd.Series(canadian_curves[wood_type+str(stress_level)])
            savings["Network_t_of_"+wood_type+str(stress_level)] = pd.Series(np.log10(t_of_cum_lists[wood_type+str(stress_level)]))
            savings["Network_curve_of_"+wood_type+str(stress_level)] = pd.Series(cum_lists[wood_type+str(stress_level)])
    savings.to_csv("Final_Performance_of_%s_%s"%(args.wood_types, args.Alter_force_scaler), index=None)
    plt.legend(loc="upper left", fontsize=8)
    plt.title("Network prediction over all types of wood and all load with mean D of %s%%"%np.round(RRs[~np.isnan(RRs)].mean(), 2))
    plt.xlabel("Logged duration of Load (log hour)")
    plt.ylabel("Failure percentage (%)")
    plt.show()

if __name__ == '__main__':

    np.random.seed(55)
    main()
