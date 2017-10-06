clear all;
clc;
disp('*** Read the dataset ***');
[filename, pathname] = uigetfile( ...
    {'*.*', 'All Files (*.*)'; ...
    }, ...
    'open data');
disp(filename)
S = [pathname filename];
sD=som_read_data(S);

k= size(sD.data,2);%nb of colons 
n=size(sD.data,1); %nb of instances

V=sD.data(:,k);
sD.data(:,1:k-1) = som_normalize(sD.data(:,1:k-1),'var');

nclasses=max(V)
mtry=round(sqrt(k-1));



    %**** Classes distribution ****
classes=unique(sD.data(:,k));
distribution(:,1)=classes;
for i=1:nclasses
    
    distribution(i,2)= size(find(sD.data(:,k)==distribution(i,1)),1)/n;
end

k_cv=5;


indice_T=[];
T=[];

indice_A=[];
A=[];
acc=[];
indice_f=zeros(k,k_cv);

indices = crossvalind('Kfold',n,k_cv);
ee=[];
ee1=[];
ee2=[];
ee3=[];
r=[];
tic
for iter=1:k_cv
itest=(indices == iter);
indice_T=find(itest==1);

T=sD.data(indice_T,:);
taille_T=length(indice_T);
indice_A=setdiff([1:n],indice_T);
taille_A=length(indice_A);
A=sD.data(indice_A,:);
YT=T(:,k);
YA=A(:,k);
YT_car=num2str(YT);   


    no_of_hypothesis=50;% nomber of trees

   
    
    classif_total=[];
    classif_A=[];
    feuille_A=[];
    M_oob=zeros(taille_A,no_of_hypothesis); 
    disp(strcat('taille_A ',num2str(taille_A)));


	for turn=1:1:no_of_hypothesis
		
        inst_rand=rand(taille_A,1)*taille_A; 
	    instances=ceil(inst_rand); 
	    bag= A(instances,:); 
	    
        
	    base=[1:taille_A]';
	    instances_unique=unique(instances);
		indiceOOBag=setdiff(base,instances_unique);
	    M_oob(indiceOOBag,turn)=1;
        oob=A(indiceOOBag,:); 
	    n_oob=size(oob,1); 
		LabelBag= bag(:,k);    
		LabelOOBag= oob(:,k);
	
	
		hi=classregtree(bag2(:,1:k-1),LabelBag, 'method','classification','prune','off','nvartosample',mtry); %arbre CART
		
		classif_total=[classif_total,str2double((eval(hi,T(:,1:k-1))))];   
        [y,nodes] = eval(hi,A(:,1:k-1));
        classif_A=[classif_A,str2double(y)];         
         H(turn)=struct('hypothese',hi,'Bag',instances,'LabelBag',LabelBag,'indiceOOBag',indiceOOBag,'LabelOOBag',LabelOOBag); 
	
    end
    
    % classify the test instances 
    classement=zeros(taille_T,nclasses); 
    for t=1:taille_T
        for c=1:nclasses
            
            classement(t,c)=size(find(classif_total(t,:)==c),2);
        end
    end   

         
    classif_total=zeros(taille_T,1);
    for t=1:taille_T
     [x,classif_total(t)]=max(classement(t,:));% recuperer l'indice de la classe majoritaire
    end      
     
    ac_bagging = mean(classif_total==YT)
    ct=length(unique(YT))
    clearvars neighbors2 stats2 predicted_labels2 neighbors stats predicted_labels
    
    % classify the oob instances
    classement=zeros(taille_A,nclasses);
    
    for t=1:taille_A
        for c=1:nclasses
            classement(t,c)=size(find(classif_A(t,find(M_oob(t,:)==1))==c),2);
            [m in] =max(classement(t,:));            
        end
    end
    
      
         %margin calculation
    marge=zeros(taille_A,1);
    for t=1:taille_A
        cc=sort(classement(t,:),'descend');
        marge(t)= (cc(1)-cc(2))/sum(cc);
    end
    mm2 id]=sort(marge2,'ascend');
    mm2=[mm2 id];
    set1=mm2(find(mm2(:,1)<0.5),2);
    set2=mm2(find(mm2(:,1)>=0.5),2);
	%select 70% of set1 and 30% of set2
	l1=length(set1)*(0.7);
	l2=length(set2)*(0.3);
selection=[set1(1:l1);set2(length(set2)-l2:length(set2))];
   

 
 
  
 end  
