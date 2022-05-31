__kernel void picalc(__global double* a, int n)
        {
        unsigned int L= get_local_size(0);
        unsigned int gid = get_global_id(0);
        double h= 1.0/(double)n;
        double sum= 0.0;
        double x;
        for(int i=gid+1;i<n;i=i+L){
            x=h*((double)i-0.5);
            sum=4.0/(1.0+x*x);
            a[i]=sum;
        }
}


