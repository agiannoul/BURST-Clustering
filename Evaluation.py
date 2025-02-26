


def purity(Predicted,Real):

    N=len(Predicted)
    un_pred=set(Predicted)
    un_real=set(Predicted)


    summ=0
    for pred in un_pred:
        #find the class that is most appeared in the cluster
        #collect positions in of cluster pred
        Real_cor_pred=[c_r for c_r, c_p in zip(Real,Predicted) if c_p == pred]
        max_same=0
        for real in un_real:
            max_same=max(max_same,len([1 for rc in Real_cor_pred if rc == real]))
        summ+=max_same
    return summ/N


