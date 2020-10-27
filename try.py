def train(gpu, args):
    print('gpu{}: Begin training'.format(gpu))
    # distributed training initialization
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12121',
        world_size=args.world_size,
        rank=gpu
    )
    torch.manual_seed(0)

    # INIT MODEL
    model = build_model(cfg)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    print("gpu{}: Finish constructing model".format(gpu))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction="none")

    # Load the Train Data
    train_cfg = cfg["train_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    print('gpu{}: Finish loading dataset'.format(gpu))

    # wrap the dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=gpu, shuffle=train_cfg["shuffle"])
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_cfg["batch_size"],
                                  num_workers=0, pin_memory=True, sampler=train_sampler)

    # wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # using apex
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O2', keep_batchnorm_fp32=True)
    # model = DDP(model)

    # loading pretrain model
    if cfg['train_params']['model_num'] != 0:
        model_path = log_dir + 'resnet_{}.pth'.format(cfg['train_params']['model_num'])
        model.load_state_dict(torch.load(model_path, map_location={'cuda:%d' % 0: 'cuda:%d' % gpu})['model'])
        print("gpu{}: Finish loading model".format(gpu))
        dist.barrier()
    else:
        print("gpu{}: No pretrain model".format(gpu))

    # TRAIN LOOP
    checkpoint = cfg['train_params']['checkpoint_every_n_steps']
    for epoch in range(cfg['train_params']['max_num_epochs']):
        print('gpu{}: Begin epoch {}'.format(gpu, epoch))
        tr_it = iter(train_dataloader)
        progress_bar = tqdm(range(len(train_dataloader)))
        losses_train = []
        for index in progress_bar:
            try:
                data = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data = next(tr_it)
            model.train()
            torch.set_grad_enabled(True)
            loss, _ = forward(data, model, gpu, criterion)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_train.append(loss.item())
            progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

            if index % checkpoint == 0 and index != 0 and gpu == 0:
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, log_dir + 'resnet_{}_{}.pth'.format(epoch, index))
    print("gpu{}: Finish training")

