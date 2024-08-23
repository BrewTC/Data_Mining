import matplotlib.pyplot as plt
import pandas as pd

def CEG_plot_new_version(file_name, y_test, test_pred):
    CEG_data = {
        'y_true': y_test,
        'y_pred': test_pred
    }

    bank = pd.DataFrame(CEG_data)
    real_col='y_true'
    pred_col='y_pred'

    #清除realBS無資料與負值與真實>600
    bank=bank.drop(bank[(bank[real_col]=='no_data')].index)
    bank=bank.drop(bank[(pd.to_numeric(bank[real_col])<40)].index) #
    bank=bank.drop(bank[(pd.to_numeric(bank[real_col])>600)].index)
    bank=bank.drop(bank[(pd.to_numeric(bank[pred_col])>600)].index) #
    #bank=bank.sort_values(['realBS'],ascending=True)
    x=list(pd.to_numeric(bank[real_col])  ) #轉為數值
    y=list(pd.to_numeric(bank[pred_col]) )

    #print(x[1])
    #計算A區準確度(醫療)
    A=0
    m=len(x)
    for i in range(0,m):
        
        diff=y[i]-x[i]
        #print(diff)
        
        if y[i]<100.1 and abs(diff)<=15:
            A+=1
        else:
            if x[i]*0.85<=y[i]<=x[i]*1.15:
                A+=1
    if m==0:
        m=9999999999  
    A_zone=A*100/m          
    print('符合A區-醫療準度:',A_zone,'%')   
    #計算A區準確度(醫療)
    A=0
    B=0
    m=len(x)
    for i in range(0,m):
        if y[i]>50 and ((x[i]*0.53)-18)<=y[i]<=((x[i]*2.466667)-71):
            A+=1
        elif y[i]<100 and abs(y[i]-x[i])<15.999:
            A+=1    
        else:
            #print('不在A/B區')
            B+=1
    if m==0:
        m=9999999999  
    AB_zone=A*100/m          
    print('符合A+B區準度:',AB_zone,'%') 
    print('不符合A+B區占比:',B*100/m,'%')  

    acc_df={"file_name":file_name,
            "total_num":m,
            "A_zone_ACC_15%":A_zone,
            "B_zone_ACC":AB_zone-A_zone,
            "A+B_zone_ACC":AB_zone,
            "out_zone":B*100/m
            }
    acc_df= pd.DataFrame(acc_df,index=[0])
    path=str(file_name)+'_'+pred_col+'_acc.csv' ############################
    acc_df.to_csv(path,index=0,header=1,encoding='utf_8_sig') #保存列名         

    plt.figure(figsize=(10,8),dpi=100)
    plt.scatter(x,y,alpha=0.5,label='Blood sugar (BS)')

    plt.xlabel('Real values')
    plt.ylabel('Predicted values') #(inlier)(Trans)

    plt.title(str(file_name)+'_pred_CEG_plot',fontsize=20)
    # plt.title(pred_col+'_CEG_plot',fontsize=20) ##########################

    plt.legend(loc='upper right')   #圖例位置
    plt.legend(loc='best')   #圖例位置
    #A區
    plt.plot([0,600],[0,600], ls='--',lw=2, color='blue')
    plt.plot([50,50],[0,35], ls='-',lw=2, color='green')
    plt.plot([0,35],[50,50], ls='-',lw=2, color='green')
    plt.plot([35,100],[50,115], ls='-',lw=2, color='green')
    plt.plot([50,100],[35,85], ls='-',lw=2, color='green')
    plt.plot([100,600],[115,690], ls='-',lw=2, color='green')
    plt.plot([100,600],[85,510], ls='-',lw=2, color='green')
    plt.text(425,475,'A:'+str(A_zone)[:5]+'%',{'fontsize':18})
    #B區
    plt.plot([120,120],[0,35], ls='-',lw=2, color='red')
    plt.plot([120,260],[35,130], ls='-',lw=2, color='red')
    plt.plot([260,600],[130,270], ls='-',lw=2, color='red')
    plt.plot([0,35],[63,63], ls='-',lw=2, color='red')
    plt.plot([35,50],[63,80], ls='-',lw=2, color='red')
    plt.plot([50,75],[80,120], ls='-',lw=2, color='red')
    plt.plot([75,310],[120,690], ls='-',lw=2, color='red')
    plt.text(425,280,'B:'+str(AB_zone-A_zone)[:5]+'%',{'fontsize':18})
    plt.text(425,100,'C+D+E:'+str(100-AB_zone)[:5]+'%',{'fontsize':18})
    
    plt.tight_layout()
    plt.savefig(str(file_name)+'CEG_plot.png', dpi=300, bbox_inches='tight')
    plt.show()