# Federated-Learning-with-Model-Quantilization

Prerequisites
-----
    Python 3.8
    Torch 1.8.0 or Tensorflow 2.3.0
Models&Data
-----
    Learning models: MLP, CNN
    Datasets: Cifar-10, FashionMNIST, MNIST, Adult
Main Parameters
-----
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--dataset', default='adult', help='mnist or FashionMNIST or cifar or Adult')
    parser.add_argument('--num_classes', default=2)
    parser.add_argument('--iid', default=True)
    parser.add_argument('--strict_iid', default=True)     
    parser.add_argument('--model', default='mlp', help='mlp or cnn')
    parser.add_argument('--hidden_units', type=list, default=[128,64], help='mlp or cnn')
    parser.add_argument('--lr', default=0.002, help='learning rate')    
    parser.add_argument('--epochs', type=int, default=3, help='number of training epochs')  
    parser.add_argument('--local_ep', type=int, default=5)     
    parser.add_argument('--num_users', type=int, default=50) 
    parser.add_argument('--num_chosenusers', type=int, default=5) 
    parser.add_argument('--num_items_train', type=int, default=800)
    parser.add_argument('--num_items_test', type=int, default=128)    
    parser.add_argument('--local_bs', type=int, default=128)
    
    parser.add_argument('--set_quant_level', type=list, default=[64])
    parser.add_argument('--set_quant_sche', type=list, \
                       default=['bucket_quantile', 'uniform_quantization', 'SVD_Split',\
                                'count_sketch', 'QSGD', 'kmeans'])
    parser.add_argument('--set_degree_noniid', type=list, default=[0])    
    parser.add_argument('--num_experiments', type=int, default=1) 
Get Started
-----
    python fedavg.py --epochs 100
    
