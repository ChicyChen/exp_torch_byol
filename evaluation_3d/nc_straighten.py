import numpy as np
import torch
import torch.nn as nn


class NC():
    def __init__(self, model, device, input_num=1):
        super(NC, self).__init__()
        self.device = device
        self.input_num = input_num
        self.model = model.to(device)
        self.model.eval()


    def extract_data(self, loader, num_batch=10):
        
        x_lst = []
        # features = []
        label_lst = []
        batch_i = 0

        with torch.no_grad():
            for data_i in loader:
                if self.input_num == 2:
                    input_tensor, _, label = data_i
                else:
                    input_tensor, label = data_i
                # h = self.model.get_representation(input_tensor.to(self.device))
                # features.append(h)
                x_lst.append(input_tensor)
                label_lst.append(label)
                batch_i += 1
                if batch_i >= num_batch:
                    break

            x_total = torch.stack(x_lst)
            # h_total = torch.stack(features)
            label_total = torch.stack(label_lst)

            # return x_total, h_total, label_total
        return x_total, label_total
        
    def seq_similarity(self, x, seq_len, num_seq):
        video_mean_list = []
        mean_list = []
        std_list = []
        var_list = []
        with torch.no_grad():
            # print(len(x))
            for i in range(len(x)):
                seq_i = x[i] # 1, C, T, H, W
                # print(seq_i.shape)
                _, C, T, H, W = seq_i.shape
                seq_i = seq_i.reshape(C, num_seq, seq_len, H, W)
                seq_i = seq_i.to(self.device)
                seq_i = seq_i.permute(1,0,2,3,4)
                # print(seq_i.size())
                out_i = self.model(seq_i).cpu().numpy() # num_seq, N
                # print(out_i.shape)

                mean_i = np.mean(out_i, axis=0)
                std_i = np.std(out_i, axis=0)
                var_i = np.var(out_i, axis=0)
                # print(mean_i.shape)
                video_mean_list.append(mean_i)
                mean_list.append(np.linalg.norm(mean_i))
                std_list.append(np.linalg.norm(std_i))
                var_list.append(np.linalg.norm(var_i))

            class_mean_mean = np.mean(mean_list)
            class_std_mean = np.mean(std_list)
            class_var_mean = np.mean(var_list)

            # print(np.mean(video_mean_list, axis=0).shape)

            cross_mean = np.linalg.norm(np.mean(video_mean_list, axis=0))
            cross_std = np.linalg.norm(np.std(video_mean_list, axis=0))
            cross_var = np.linalg.norm(np.var(video_mean_list, axis=0))

        
        return class_mean_mean, class_std_mean, class_var_mean, cross_mean, cross_std, cross_var
    
    def seq_linear(self, x, seq_len, num_seq):
        c_seq_list = []
        c_out_list = []

        with torch.no_grad():
            # print(len(x))
            for i in range(len(x)):
                seq_i = x[i] # 1, C, T, H, W
                # print(seq_i.shape)
                _, C, T, H, W = seq_i.shape
                seq_i = seq_i.reshape(C, num_seq, seq_len, H, W)
                seq_i = seq_i.to(self.device)
                seq_i = seq_i.permute(1,0,2,3,4) # num_seq, C, seq_len, H, W
                seq_diff_i = (seq_i[1:] - seq_i[:-1]).cpu().numpy().reshape(num_seq-1, -1)
                # seq_diff_i = seq_diff_i
                # print(seq_diff_i.shape)
                out_i = self.model(seq_i).cpu().numpy() # num_seq, N
                out_diff_i = out_i[1:] - out_i[:-1]
                # out_diff_i = out_diff_i
                # print(out_diff_i.shape)
                c_seq_i = 0.0
                c_out_i = 0.0
                for j in range(num_seq - 2):
                    c_seq_j = np.arccos(np.sum(seq_diff_i[j]/np.linalg.norm(seq_diff_i[j]) * seq_diff_i[j+1]/np.linalg.norm(seq_diff_i[j+1])))
                    c_seq_i += c_seq_j / (num_seq - 2)
                    c_out_j = np.arccos(np.sum(out_diff_i[j]/np.linalg.norm(out_diff_i[j]) * out_diff_i[j+1]/np.linalg.norm(out_diff_i[j+1])))
                    c_out_i += c_out_j / (num_seq - 2)
                c_seq_list.append(c_seq_i)
                c_out_list.append(c_out_i)

            c_seq_mean = np.mean(c_seq_list)
            c_out_mean = np.mean(c_out_list)
            # print(np.arccos(-1))
        return c_seq_mean, c_out_mean
    
    def seq_pca_linear(self, x, seq_len, num_seq, proj_dim=3):
        seq_list = []
        out_list = []
        c_seq_list = []
        c_out_list = []

        with torch.no_grad():
            # print(len(x))
            for i in range(len(x)):
                seq_i = x[i] # 1, C, T, H, W
                # print(seq_i.shape)
                _, C, T, H, W = seq_i.shape
                seq_i = seq_i.reshape(C, num_seq, seq_len, H, W)
                seq_i = seq_i.to(self.device)
                seq_i = seq_i.permute(1,0,2,3,4) # num_seq, C, seq_len, H, W

                seq_i_flatten = seq_i.reshape(num_seq,-1) # num_seq, D
                seq_list.append(seq_i_flatten)
                # seq_diff_i = (seq_i_flatten[1:] - seq_i_flatten[:-1]).cpu().numpy()
                # U,S,V = torch.pca_lowrank(seq_i_flatten, q=None, center=True, niter=3)
                # seq_i_pca=torch.matmul(seq_i_flatten, V[:, :proj_dim])
                # seq_diff_i = (seq_i_pca[1:] - seq_i_pca[:-1]).cpu().numpy().reshape(num_seq-1, -1)
                # print(seq_diff_i.shape)

                out_i = self.model(seq_i) # num_seq, N
                out_list.append(out_i)
                # out_diff_i = (out_i[1:] - out_i[:-1]).cpu().numpy()
                # U,S,V = torch.pca_lowrank(out_i, q=None, center=True, niter=3)
                # out_i_pca=torch.matmul(out_i, V[:, :proj_dim])
                # out_diff_i = (out_i_pca[1:] - out_i_pca[:-1]).cpu().numpy()
                # print(out_diff_i.shape)

            seq_total = torch.stack(seq_list).reshape(len(x)*num_seq, -1)
            out_total = torch.stack(out_list).reshape(len(x)*num_seq, -1) # len(x)*num_seq, N

            U,S,V = torch.pca_lowrank(seq_total, q=None, center=True, niter=3) # V: *, N
            seq_total_pca=torch.matmul(seq_total, V[:, :proj_dim]).cpu().numpy() # len(x)*num_seq, proj_dim
            U,S,V = torch.pca_lowrank(out_total, q=None, center=True, niter=3) # V: *, N
            out_total_pca=torch.matmul(out_total, V[:, :proj_dim]).cpu().numpy() # len(x)*num_seq, proj_dim

            for i in range(len(x)):
                c_seq_i = 0.0
                c_out_i = 0.0
                seq_i_flatten = seq_total_pca[i*num_seq:(i+1)*num_seq]
                out_i=out_total_pca[i*num_seq:(i+1)*num_seq] # num_seq, proj_dim
                
                seq_diff_i = seq_i_flatten[1:] - seq_i_flatten[:-1]
                out_diff_i = out_i[1:] - out_i[:-1]

                for j in range(num_seq - 2):
                    c_seq_j = np.arccos(np.sum(seq_diff_i[j]/np.linalg.norm(seq_diff_i[j]) * seq_diff_i[j+1]/np.linalg.norm(seq_diff_i[j+1])))
                    c_seq_i += c_seq_j / (num_seq - 2)
                    c_out_j = np.arccos(np.sum(out_diff_i[j]/np.linalg.norm(out_diff_i[j]) * out_diff_i[j+1]/np.linalg.norm(out_diff_i[j+1])))
                    c_out_i += c_out_j / (num_seq - 2)
                c_seq_list.append(c_seq_i)
                c_out_list.append(c_out_i)

            c_seq_mean = np.mean(c_seq_list)
            c_out_mean = np.mean(c_out_list)
            # print(np.arccos(-1))
        return c_seq_mean, c_out_mean

    def visualize_latent_pca(self, x, labels, seq_len, num_seq, fname):
        from mpl_toolkits import mplot3d
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        seq_list = []
        out_list = []

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        colors = cm.rainbow(np.linspace(0, 1, len(x)))

        with torch.no_grad():
            # print(len(x))
            for i in range(len(x)):
                seq_i = x[i] # 1, C, T, H, W
                # print(seq_i.shape)
                _, C, T, H, W = seq_i.shape
                seq_i = seq_i.reshape(C, num_seq, seq_len, H, W)
                seq_i = seq_i.to(self.device)
                seq_i = seq_i.permute(1,0,2,3,4) # num_seq, C, seq_len, H, W
                out_i = self.model(seq_i) # num_seq, N

                seq_list.append(seq_i)
                out_list.append(out_i)

            # seq_total = torch.stack(seq_list).reshape(len(x)*num_seq, C*seq_len*H*W)
            out_total = torch.stack(out_list).reshape(len(x)*num_seq, -1) # len(x)*num_seq, N

            U,S,V = torch.pca_lowrank(out_total, q=None, center=True, niter=3) # V: *, N
            out_total_pca=torch.matmul(out_total, V[:, :3]).cpu().numpy() # len(x)*num_seq, 3

            for i in range(len(x)):
                c = colors[i]
                out_i_pca=out_total_pca[i*num_seq:(i+1)*num_seq] # num_seq, 3
                ax.scatter3D(out_i_pca[:,0], out_i_pca[:,1], out_i_pca[:,2], color=c)
                ax.plot3D(out_i_pca[:,0], out_i_pca[:,1], out_i_pca[:,2], color=c)
                

            plt.savefig(fname)
        return
    

    def visualize_class(self, x, labels, seq_len, num_seq, fname, num_class=101):
        from mpl_toolkits import mplot3d
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        seq_list = []
        out_list = []

        plt.figure()
        # ax = plt.axes(projection='3d')
        colors = cm.rainbow(np.linspace(0, 1, num_class))

        with torch.no_grad():
            # print(len(x))
            for i in range(len(x)):
                seq_i = x[i] # B = 1, C, T, H, W
                # print(seq_i.shape)
                _, C, T, H, W = seq_i.shape
                seq_i = seq_i.reshape(C, num_seq, seq_len, H, W)
                seq_i = seq_i.to(self.device)
                seq_i = seq_i.permute(1,0,2,3,4) # num_seq, C, seq_len, H, W
                out_i = self.model(seq_i) # num_seq, N
                # print(out_i.shape)

                # seq_list.append(np.mean(seq_i, axis=0))
                mean_i = torch.mean(out_i, axis=0)
                # print(mean_i.size())
                out_list.append(torch.mean(out_i, axis=0))
                

            # seq_total = torch.stack(seq_list).reshape(len(x), C*seq_len*H*W)
            out_total = torch.stack(out_list).reshape(len(x), -1) # len(x)*num_seq, N
            # print(out_total.shape)

            U,S,V = torch.pca_lowrank(out_total, q=10, center=True, niter=3) # V: *, N
            print(S)
            out_total_pca=torch.matmul(out_total, V[:, :3]).cpu().numpy() # len(x)*num_seq, 3
            # print(out_total_pca.shape)

            for i in range(len(x)):
                c = colors[labels[i]]
                # print(labels[i])
                # out_i_pca=out_total_pca[i*num_seq:(i+1)*num_seq] # num_seq, 3
                out_i_pca=out_total_pca[i]
                # ax.scatter3D(out_i_pca[0], out_i_pca[1], out_i_pca[2], color=c)
                plt.scatter(out_i_pca[0], out_i_pca[1], color=c)
                # ax.plot3D(out_i_pca[:,0], out_i_pca[:,1], out_i_pca[:,2], color=c)

            ax = plt.gca()
            ax.set_aspect('equal')
            plt.savefig(fname)
        return

                