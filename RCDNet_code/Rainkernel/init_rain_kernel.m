%% initialization rain kernel
num = 32;
C = zeros(num,9,9);
num1 = 12;
    for i  = 1:num1
        C( i, :,:) = MyRainK(9, 2.5+3/12*i, 0.1, 0.25);
        figure(1)
        tu_da = imresize(permute(C(i,:,:),[2,3,1]),10,'nearest');
        imshow(tu_da,[]);
        disp(sum(C(:)));
        disp(max(C(:)));
        pause(0.1)
    end
    for i  = num1+1:num
        C( i, :,:) = MyRainK(9, 2+3/12*i, 0.2, 0.5);
        figure(1)
        tu_da = imresize(permute(C(i,:,:),[2,3,1]),10,'nearest');
        imshow(tu_da,[]);
        disp(sum(C(:)));
        disp(max(C(:)));
        pause(0.1)
    end
C = bsxfun(@times, permute(C,[4,1,2,3]),ones([3,1]));
C9 = squeeze(C(:,:,:,:));
save('init_kernel','C9');  % 9 means the rain kernel size 9*9
