# load encoder
weights = R3D_18_Weights.DEFAULT
preprocess = weights.transforms()

trained_encoder = r3d_18()
fc_layer = torch.nn.Linear(trained_encoder.fc.in_features, 3)

trained_encoder = torch.nn.Sequential(*(list(trained_encoder.children())[:-1]))
trained_encoder.load_state_dict(torch.load(f'/path to the saved encoder/best_encoder_ScratchLearn.pth', weights_only=True))

# dont require gradient for encoder
for param in trained_encoder.parameters():
    param.requires_grad = False
    
for param in fc_layer.parameters():
    param.requires_grad = True

# eval mode
trained_encoder.eval()

# Define loss function and optimizer
CELoss = nn.CrossEntropyLoss()
optimizer_fc = optim.Adam(fc_layer.parameters(), lr=1e-4) 

num_epochs = 400
trained_encoder = trained_encoder.to(device1)
fc_layer = fc_layer.to(device1)
fc_layer.train()

BatchSize = 64

results_file = open('/path to the training log/training_log.txt', 'a')
best_accuracy = 0

for epoch in range(num_epochs):
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
            with torch.no_grad():
                anchor_data = trained_encoder(data)
                anchor_data = anchor_data.view(anchor_data.size(0), -1)

            pred = fc_layer(anchor_data)
            supervised_loss = CELoss(pred, targets)
            # backward
            optimizer_fc.zero_grad()
            supervised_loss.backward()
            optimizer_fc.step()
            running_loss += supervised_loss
        
        del train_loader, data, targets
    
    # Validation phase
    total_val_correct = 0
    total_val = 0
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(device1)
            batch_Y = batch_Y.to(device1)
            h_val = trained_encoder(batch_X)
            h_val = h_val.view(h_val.size(0), -1)
            val_scores = fc_layer(h_val).softmax(-1)  # Forward pass on validation set
            _, val_predicted = torch.max(val_scores.data, 1)  # Get predictions
            total_val_correct += (val_predicted == batch_Y).sum().item()  # Count correct predictions
            total_val += batch_Y.size(0)  # Total predictions
            val_loss += CELoss(fc_layer(h_val), batch_Y)

    val_accuracy = total_val_correct / total_val  # Calculate validation accuracy
    results_file.write(f'Epoch [{epoch+1}/{num_epochs}], Training loss: {running_loss.item():.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.3f}\n')
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(fc_layer.state_dict(), f'/path to save the fc layer/best_fc_layer_ScratchLearn.pth')
        results_file.write(f'New best fc_layer saved with val accuracy: {val_accuracy}\n')
    torch.cuda.empty_cache()

results_file.close()
