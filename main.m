%% Subject 1

clc
clear

%% Load Data
subject = 1;
channels = [23, 23, 23, 28, 28];
ch = channels(subject);

% With Epilesy
folderPath = num2str("/Users/parniantaheri/Desktop/temp/class/term 7/یادگیری در مغز و ماشین ۱/HW/HW4/dataset/Sub"+ subject+ "/WS"+ subject+ "/");

if subject<10
file_name = num2str("chb0"+subject+"_");
else
file_name =  num2str("chb"+subject+"_");
end
ws_files = read_file(folderPath, file_name);

ws_start = [2296 ,1467 ,1732 ,1015, 1720, 1862];

% Without Epilesy
folderPath = num2str("/Users/parniantaheri/Desktop/temp/class/term 7/یادگیری در مغز و ماشین ۱/HW/HW4/dataset/Sub"+ subject+ "/WOS"+ subject+ "/");

if subject<10
file_name = num2str("chb0"+subject+"_");
else
file_name =  num2str("chb"+subject+"_");
end
wos_files = read_file(folderPath, file_name);

%% Time Selection
ws = time_selection(ws_start, ws_files);

wos_random_time = wos_choose_time(ws_start, wos_files);
wos = time_selection(wos_random_time, wos_files);

%% Feature Extraction
ws_feature = featureExtraction(ws, ch);
wos_feature = featureExtraction(wos, ch);

%% Split
l = size(ws_feature, 1);
train_num = round(0.7 * l);
train_neg_idx = randperm(l, train_num);
train_pos_idx = randperm(l, train_num);

train_neg = wos_feature(train_neg_idx, :);
train_pos = ws_feature(train_pos_idx, :);

test_neg = wos_feature(setdiff(1:l, train_neg_idx), :);
test_pos = ws_feature(setdiff(1:l, train_pos_idx), :);

ws_train_labels = ones(size(train_pos,1), 1);
wos_train_labels = zeros(size(train_pos,1), 1);
ws_test_labels = ones(size(test_pos,1), 1);
wos_test_labels = zeros(size(test_pos,1), 1);


%% Test Features
[x_train,x_test] = feature_selection(train_pos,train_neg, test_pos, test_neg);

y_train = [ws_train_labels; wos_train_labels];

y_test = [ws_test_labels; wos_test_labels];
%% Classification
% KNN
knnClassifier = fitcknn(x_train, y_train, 'NumNeighbors', 2); 
predictions_knn = predict(knnClassifier, x_test);
accuracy_knn = sum(predictions_knn == y_test) / length(y_test)*100;
[knn_sensitivity, knn_specificity] = sensitivity_specificity(y_test, predictions_knn);

%SVM
svmClassifier = fitcsvm(x_train, y_train, 'KernelFunction', 'linear');
predictions_svm = predict(svmClassifier, x_test);
accuracy_svm = sum(predictions_svm == y_test) / length(y_test)*100;
[svm_sensitivity, svm_specificity] = sensitivity_specificity(y_test, predictions_svm);

%% Results
disp("")
disp(["Results for Subject "+subject])
disp("----------------------------------------------------------------------")
disp("|   Classifier   |    Accuracy   |   Sensitivity   |   Specificity   |")
disp("----------------------------------------------------------------------")
fprintf("|       KNN      |       %d      |       %.2f      |      %.2f       |\n",accuracy_knn, knn_sensitivity, knn_specificity)
fprintf("|       SVM      |       %d      |       %.2f      |      %.2f       |\n\n\n",accuracy_svm, svm_sensitivity, svm_specificity)
%% Functions

function ws_file = read_file(FolderPath, file_name)
    edfFiles = dir(fullfile(FolderPath, [file_name, '*.edf']));
    ws_file = cell(1, numel(edfFiles));
    for i = 1:numel(edfFiles)
        filePath = fullfile(FolderPath, edfFiles(i).name);
        ws_file{i} = edfread(filePath);
    end
end


function selected_time = time_selection(start_points, files)
    l = length(start_points); 
    % dividing 9 min to 16 sec intervals
    epoches = 34;
    ws = cell(l, epoches);
    sec = 16;
    for j=1:l
        for i=1:epoches
            startTime = seconds(start_points(j) - i .* sec);
            endTime = seconds(start_points(j) - (i-1) .* sec);
            timeRange = timerange(startTime, endTime);
            % select the jth file
            if size(files,2) == 1
                file = files;
            else
                file = files(j);
            end
            file = file{1};
            ws{j,i} = table2array(file(timeRange, :));
        end
    end
    % because the start time was later than the stop tipe we flip it
    selected_time = flip(ws,2);
end


function random_time = wos_choose_time(ws_start, wos_files)
    l = length(ws_start);
    timetableLength = size(wos_files{1},1);

    % To go backward for 9 min
    start = 540;
    stop = timetableLength;

    % Generate unique random numbers
    random_time = randperm(stop - start, l) + start;
end


function total_features = featureExtraction(data, channel)
    disp("Start Feature Extraction")
    l = size(data, 1);
    epoches = 34;
    total_num_feature = epoches * 5 * channel;

    features = zeros(l,total_num_feature);
    total_features = zeros(l, total_num_feature);

    for k=1:l
        cnt = 0;
        for j=1:epoches
             temp = data(k,j);
             temp = temp{1};
            for i=1:channel
                ch_i = temp(:,i);

                % vectorize
                concatenatedArray = vertcat(ch_i{:});

                % STD
                cnt = cnt + 1;
                features(k,cnt) = std(concatenatedArray);
    
                % Mean
                cnt = cnt + 1;
                features(k,cnt) = mean(concatenatedArray);
                
                % Min
                cnt = cnt + 1;
                features(k,cnt) = min(concatenatedArray);
                
                % Max
                cnt = cnt + 1;
                features(k,cnt) = max(concatenatedArray);

                % Shannon Entropy
                cnt = cnt + 1;
                fs = 256;
                [Pxx, f] = periodogram(concatenatedArray, [], [], fs);
                norm_Pxx = Pxx / max(Pxx);
                features(k,cnt) = -sum(norm_Pxx .* log2(norm_Pxx));
               
            end
        end
        total_features(k,:) = features(k,:);
    end 
end


function [x_train,x_test] = feature_selection(train_pos,train_neg, test_pos, test_neg)
    train = [train_pos;train_neg];
    test = [test_pos;test_neg];

    p_values = zeros(1, size(train, 2));
    for i = 1:size(train, 2)
    [~, p_values(i)] = ttest(train(:, i));
    end

    alpha = 0.001;
    significant_features = find(p_values < alpha);
    x_train = train(:, significant_features);
    x_test = test(:, significant_features);
end


function [sensitivity, specificity] = sensitivity_specificity(y, predictions)
    confusionMatrix = confusionmat(y, predictions);
    TP = confusionMatrix(1, 1);
    FP = confusionMatrix(2, 1);
    TN = confusionMatrix(2, 2);
    FN = confusionMatrix(1, 2);
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
end



