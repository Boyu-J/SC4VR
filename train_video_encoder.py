# load model
weights = R3D_18_Weights.DEFAULT
encoder_anchor = r3d_18()

# drop the classification head
encoder_anchor = torch.nn.Sequential(*(list(encoder_anchor.children())[:-1]))

# require gradient for training
for param in encoder_anchor.parameters():
    param.requires_grad = True

# Initialize the video transforms
weights = R3D_18_Weights.DEFAULT
preprocess = weights.transforms()

# load validation sets
with ThreadPoolExecutor() as executor:
    X_val = list(executor.map(process_video, val_files))

X_val = torch.stack(X_val)#.to(device1)
Y_val = torch.as_tensor(val_labels)#.to(device1)

# Create a DataLoader for batching
val_dataset = TensorDataset(X_val, Y_val)
del X_val, Y_val
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)  # Adjust batch_size as needed

num_epochs = 400
encoder_anchor = encoder_anchor.to(device1)
encoder_anchor.train()
BatchSize = 64



# Define loss function and optimizer
ContrastLoss = ContrastiveLoss()
optimizer_encoder = optim.Adam(params = encoder_anchor.parameters(), lr=1e-4)


best_val_ss = 0
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=num_epochs, eta_min=0)

for epoch in range(num_epochs):
    results_file = open('/path to training log/training_log.txt', 'a')
    running_loss = 0
    for split in range(num_splits):
        # Process each split
        with ThreadPoolExecutor() as executor:
            X_split = list(executor.map(process_video, train_files_split[split]))
            
        X_split = torch.stack(X_split)
        Y_split = torch.as_tensor(train_labels_split[split])
        train_dataset = TensorDataset(X_split, Y_split)
        del X_split, Y_split
        train_loader = DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True)
        del train_dataset
   
        for data, targets in train_loader:
            # Get data to cuda
            data = data.to(device1)
            targets = targets.to(device1)
            # forward
            h = encoder_anchor(data)
            h = h.view(h.size(0), -1)
            contrastive_loss = ContrastLoss(h, targets)
            running_loss += contrastive_loss
            
            # backward
            optimizer_encoder.zero_grad()
            contrastive_loss.backward()#retain_graph=True)
            optimizer_encoder.step()
            
        
        del train_loader, data, targets

    # Validation phase
    ss_result = []
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(device1)
            batch_Y = batch_Y.to(device1)
            h_val = encoder_anchor(batch_X)
            h_val = h_val.view(h_val.size(0), -1)
            val_loss += ContrastLoss(h_val, batch_Y)
            ss = silhouette_score(h_val.cpu().numpy(), batch_Y.cpu().numpy())
            ss_result.append(ss)

    mean_ss = sum(ss_result) / len(ss_result)
    results_file.write(f'Epoch [{epoch+1}/{num_epochs}], Contrastive Loss: {running_loss.item():.5f}, Val loss: {val_loss:.5f}, Val Silht score: {mean_ss:.5f}\n')
    
    if mean_ss > best_val_ss:
        best_val_ss = mean_ss
        torch.save(encoder_anchor.state_dict(), f'/path to save the encoder/best_encoder_ScratchLearn.pth')
        results_file.write(f'New best encoder saved with Silht scores: {best_val_ss}\n')
    
    torch.cuda.empty_cache()
    results_file.close()
    scheduler.step()
