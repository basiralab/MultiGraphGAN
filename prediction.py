import os
import time
import datetime
import itertools
import torch
import SIMLR
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from model import GCNencoder, GCNdecoder
from model import Discriminator
from data_loader import *
from centrality import *
import numpy as np

class MultiGraphGAN(object):
    """
    Build MultiGraphGAN model for training and testing.
    """
    def __init__(self, src_loader, tgt_loaders, nb_clusters, opts):
        self.src_loader = src_loader
        self.tgt_loaders = tgt_loaders
        self.opts = opts
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # criterion function
        self.criterionIdt = torch.nn.L1Loss()
        # build models
        self.build_model()
        self.build_generators(nb_clusters)
        self.nb_clusters = nb_clusters

    def build_model(self):
        """
        Build encoder and discriminator models and initialize optimizers.
        """
        # build shared encoder
        self.E = GCNencoder(self.opts.in_feature, self.opts.hidden1, self.opts.hidden2, self.opts.dropout).to(self.device)
        
        # build discriminator( combined with the auxiliary classifier )
        self.D = Discriminator(self.opts.in_feature, 1, self.opts.dropout).to(self.device)
       
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.opts.d_lr, [self.opts.beta1, self.opts.beta2])

    def build_generators(self,nb_clusters):
        """
        Build cluster-specific generators models and initialize optimizers.
        """
        self.Gs = []
        param = []
        for i in range(self.opts.num_domains - 1):
            inside_list=[]
            for i in range (nb_clusters):
                G_i = GCNdecoder(self.opts.hidden2, self.opts.hidden1, self.opts.in_feature, self.opts.dropout).to(self.device)
                inside_list.append(G_i)
                param.append(G_i)
            self.Gs.append(inside_list)
            
        # build optimizers
        param_list = [self.E.parameters()] + [G.parameters() for G in param]
        self.g_optimizer = torch.optim.Adam(itertools.chain(*param_list),
                                            self.opts.g_lr, [self.opts.beta1, self.opts.beta2])

    def restore_model(self, resume_iters, nb_clusters):
        """
        Restore the trained generators and discriminator.
        """
        print('Loading the trained models from step {}...'.format(resume_iters))

        E_path = os.path.join(self.opts.checkpoint_dir, '{}-E.ckpt'.format(resume_iters))
        self.E.load_state_dict(torch.load(E_path, map_location=lambda storage, loc: storage))

        for c in range(nb_clusters):
            for i in range(self.opts.num_domains - 1):
                G_i_path = os.path.join(self.opts.checkpoint_dir, '{}-G{}-{}.ckpt'.format(resume_iters, i+1, c))
                print(G_i_path )
                self.Gs[i][c].load_state_dict(torch.load(G_i_path, map_location=lambda storage, loc: storage))

        D_path = os.path.join(self.opts.checkpoint_dir, '{}-D.ckpt'.format(resume_iters))
        if os.path.exists(D_path):
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))


    def reset_grad(self):
        """
        Reset the gradient buffers.
        """
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()


    def gradient_penalty(self, y, x, Lf):
        """
        Compute gradient penalty.
        """
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))

        ZERO = torch.zeros_like(dydx_l2norm).to(self.device)
        penalty = torch.max(dydx_l2norm - Lf, ZERO)
        return torch.mean(penalty) ** 2


    def classification_loss(self, logit, target, type='LS'):
        """
        Compute classification loss.
        """
        print(type)
        if type == 'BCE':
            return F.binary_cross_entropy_with_logits(logit, target)
        elif type == 'LS':
            return F.mse_loss(logit, target)
        else:
            assert False, '[*] classification loss not implemented.'


    def train(self):
        """
        Train MultiGraphGAN
        """
        nb_clusters = self.nb_clusters
        
        #fixed data for evaluating: generate samples.
        src_iter = iter(self.src_loader)
        
        x_src_fixed= next(src_iter)
        x_src_fixed = x_src_fixed[0].to(self.device)
        d = next(iter(self.src_loader))
        
        tgt_iters = []
        for loader in self.tgt_loaders:
            tgt_iters.append(iter(loader))

        # label
        label_pos = torch.FloatTensor([1] * d[0].shape[0]).to(self.device)
        label_neg = torch.FloatTensor([0] * d[0].shape[0]).to(self.device)
        
        # Start training from scratch or resume training.
        start_iters = 0
        if self.opts.resume_iters:
            start_iters = self.opts.resume_iters
            self.restore_model(self.opts.resume_iters)

        # Start training.
        print('Start training MultiGraphGAN...')
        start_time = time.time()
        
        for i in range(start_iters, self.opts.num_iters):
            print("iteration",i)
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            try:
                x_src = next(src_iter)
            except:
                src_iter = iter(self.src_loader)
                x_src = next(src_iter)

            x_src = x_src[0].to(self.device)

            x_tgts = []
            for tgt_idx in range(len(tgt_iters)):
                try:
                    x_tgt_i= next(tgt_iters[tgt_idx])
                    x_tgts.append(x_tgt_i)
                except:
                    tgt_iters[tgt_idx] = iter(self.tgt_loaders[tgt_idx])
                    x_tgt_i= next(tgt_iters[tgt_idx])
                    x_tgts.append(x_tgt_i)

            for tgt_idx in range(len(x_tgts)):
                x_tgts[tgt_idx] = x_tgts[tgt_idx][0].to(self.device)
                print("x_tgts",x_tgts[tgt_idx].shape)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            
            embedding = self.E(x_src,learn_adj(x_src)).detach()
            ## Cluster the source graph embeddings using SIMLR
            simlr = SIMLR.SIMLR_LARGE(nb_clusters, embedding.shape[0]/2, 0)
            S, ff, val, ind = simlr.fit(embedding)
            y_pred = simlr.fast_minibatch_kmeans(ff,nb_clusters)
            y_pred = y_pred.tolist()
            get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

            x_fake_list = []
            x_src_list = []
            d_loss_cls = 0
            d_loss_fake = 0
            d_loss = 0

            print("Train the discriminator")
            for par in range(nb_clusters):
                print("================")
                print("cluster",par)
                print("================")
                cluster_index_list = get_indexes(par,y_pred)
                print(cluster_index_list)
                for idx in range(len(self.Gs)):
                    x_fake_i = self.Gs[idx][par](embedding[cluster_index_list],learn_adj(x_tgts[idx][cluster_index_list])).detach()
                    x_fake_list.append(x_fake_i)
                    x_src_list.append(x_src[cluster_index_list])

                    out_fake_i, out_cls_fake_i = self.D(x_fake_i,learn_adj(x_fake_i))
                    _, out_cls_real_i = self.D(x_tgts[idx][cluster_index_list],learn_adj(x_tgts[idx][cluster_index_list]))

                    ### Graph domain classification loss
                    d_loss_cls_i = self.classification_loss(out_cls_real_i, label_pos[cluster_index_list], type=self.opts.cls_loss) \
                                   + self.classification_loss(out_cls_fake_i, label_neg[cluster_index_list], type=self.opts.cls_loss)
                    d_loss_cls += d_loss_cls_i

                    # Part of adversarial loss
                    d_loss_fake += torch.mean(out_fake_i)

                out_src, out_cls_src = self.D(x_src[cluster_index_list],learn_adj(x_src[cluster_index_list]))
                ### Adversarial loss
                d_loss_adv = torch.mean(out_src) - d_loss_fake / (self.opts.num_domains - 1)

                ### Gradient penalty loss
                x_fake_cat = torch.cat(x_fake_list)
                x_src_cat = torch.cat(x_src_list)

                alpha = torch.rand(x_src_cat.size(0), 1).to(self.device)
                x_hat = (alpha * x_src_cat.data + (1 - alpha) * x_fake_cat.data).requires_grad_(True)

                out_hat, _ = self.D(x_hat,learn_adj(x_hat.detach()))
                d_loss_reg = self.gradient_penalty(out_hat, x_hat, self.opts.Lf)

                # Cluster-based loss to update the discriminator
                d_loss_cluster = -1 * d_loss_adv + self.opts.lambda_cls * d_loss_cls + self.opts.lambda_reg * d_loss_reg
                
                ### Discriminator loss
                d_loss += d_loss_cluster


            print("d_loss",d_loss)
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_adv'] = d_loss_adv.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_reg'] = d_loss_reg.item()

            # =================================================================================== #
            #                       3. Train the cluster-specific generators                      #
            # =================================================================================== #
            print("Train the generators")
            if (i + 1) % self.opts.n_critic == 0:
                g_loss_info = 0
                g_loss_adv = 0
                g_loss_idt = 0
                g_loss_topo = 0
                g_loss_rec = 0
                g_loss = 0
                
                for par in range(nb_clusters):
                    print("cluster",par)
                    for idx in range(len(self.Gs)):
                        # ========================= #
                        # =====source-to-target==== #
                        # ========================= #
                        x_fake_i = self.Gs[idx][par](embedding[cluster_index_list],learn_adj(x_tgts[idx][cluster_index_list]))
                        
                        # Global topology loss
                        global_topology = self.criterionIdt(x_fake_i, x_tgts[idx][cluster_index_list])
                        
                        # Local topology loss
                        real_topology = topological_measures(x_tgts[idx][cluster_index_list])
                        fake_topology = topological_measures(x_fake_i.detach())
                        # 0:closeness centrality    1:betweeness centrality    2:eginvector centrality
                        local_topology = mean_absolute_error(fake_topology[0],real_topology[0])
                        
                        ### Topology loss
                        g_loss_topo += (local_topology + global_topology)
                        
                        if self.opts.lambda_idt > 0:
                            x_fake_i_idt = self.Gs[idx][par](self.E(x_tgts[idx][cluster_index_list],learn_adj(x_tgts[idx][cluster_index_list])),learn_adj(x_tgts[idx][cluster_index_list]))
                            g_loss_idt += self.criterionIdt(x_fake_i_idt, x_tgts[idx][cluster_index_list])

                        out_fake_i, out_cls_fake_i = self.D(x_fake_i,learn_adj(x_fake_i.detach()))

                        ### Information maximization loss
                        g_loss_info_i = F.binary_cross_entropy_with_logits(out_cls_fake_i, label_pos[cluster_index_list])
                        g_loss_info += g_loss_info_i

                        ### Adversarial loss
                        g_loss_adv -= torch.mean(out_fake_i) # opposed sign

                        # ========================= #
                        # =====target-to-source==== #
                        # ========================= #
                        x_reconst = self.Gs[idx][par](self.E(x_fake_i,learn_adj(x_fake_i.detach())),learn_adj(x_fake_i.detach()))
                        
                        # Reconstructed global topology loss
                        reconstructed_global_topology = self.criterionIdt(x_src[cluster_index_list], x_reconst)
                        
                        # Reconstructed local topology loss
                        real_topology = topological_measures(x_src[cluster_index_list])
                        fake_topology = topological_measures(x_reconst.detach())
                        # 0:closeness centrality    1:betweeness centrality    2:eginvector centrality
                        reconstructed_local_topology = mean_absolute_error(fake_topology[0],real_topology[0])
                        
                        ### Graph reconstruction loss
                        g_loss_rec += (reconstructed_local_topology + reconstructed_global_topology)

                    # Cluster-based loss to update the generators
                    g_loss_cluster = g_loss_adv / (self.opts.num_domains - 1) + self.opts.lambda_info * g_loss_info + self.opts.lambda_idt * g_loss_idt + self.opts.lambda_topology * g_loss_topo + self.opts.lambda_rec * g_loss_rec
                    
                    ### Generator loss
                    g_loss += g_loss_cluster

                print("g_loss",g_loss)
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_adv'] = g_loss_adv.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_info.item()
                if self.opts.lambda_idt > 0:
                    loss['G/loss_idt'] = g_loss_idt.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            # print out training information.
            if (i + 1) % self.opts.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.opts.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

           
            # save model checkpoints.
            if (i + 1) % self.opts.model_save_step == 0:
              E_path = os.path.join(self.opts.checkpoint_dir, '{}-E.ckpt'.format(i+1))
              torch.save(self.E.state_dict(), E_path)

              D_path = os.path.join(self.opts.checkpoint_dir, '{}-D.ckpt'.format(i+1))
              torch.save(self.D.state_dict(), D_path)

              for par in range(nb_clusters):
                  for idx in range(len(self.Gs)):
                      G_i_path = os.path.join(self.opts.checkpoint_dir, '{}-G{}-{}.ckpt'.format(i+1, idx+1, par))
                      print(G_i_path)
                      torch.save(self.Gs[idx][par].state_dict(), G_i_path)

              print('Saved model checkpoints into {}...'.format(self.opts.checkpoint_dir))
             
              print('=============================')
              print("End of Training")
              print('=============================')

    # =================================================================================== #
    #                              5. Test with a new dataset                             #
    # =================================================================================== #
    def test(self):
        """
        Test the trained MultiGraphGAN.
        """
        self.restore_model(self.opts.test_iters,self.opts.nb_clusters)
        
        # Set data loader.
        src_loader = self.src_loader
        x_src = next(iter(self.src_loader))
        x_src = x_src[0].to(self.device)
        
        tgt_iters = []
        for loader in self.tgt_loaders:
            tgt_iters.append(iter(loader))
        
        x_tgts = []
        for tgt_idx in range(len(tgt_iters)):
            try:
                x_tgt_i= next(tgt_iters[tgt_idx])
                x_tgts.append(x_tgt_i)
            except:
                tgt_iters[tgt_idx] = iter(self.tgt_loaders[tgt_idx])
                x_tgt_i= next(tgt_iters[tgt_idx])
                x_tgts.append(x_tgt_i)

        for tgt_idx in range(len(x_tgts)):
            x_tgts[tgt_idx] = x_tgts[tgt_idx][0].to(self.device)
           
        # return model.eval()
        for par in range(self.opts.nb_clusters):
            for idx in range(len(self.Gs)):
                self.Gs[idx][par].eval()

        with torch.no_grad():
            embedding = self.E(x_src,learn_adj(x_src))
            predicted_target_graphs = []
            for idx in range(len(self.Gs)):
                sum_cluster_pred_graph = 0
                for par in range(self.opts.nb_clusters):
                    x_fake_i = self.Gs[idx][par](embedding,learn_adj(x_src))
                    sum_cluster_pred_graph = np.add(sum_cluster_pred_graph,x_fake_i)

                average_predicted_target_graph = sum_cluster_pred_graph / float(self.opts.nb_clusters)
                predicted_target_graphs.append(average_predicted_target_graph)
                
        
        print('=============================')
        print("End of Testing")
        print('=============================')
        
        return predicted_target_graphs, x_src