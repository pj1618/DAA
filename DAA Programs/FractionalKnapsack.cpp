#include<iostream>
#include<bits/stdc++.h>
using namespace std;
struct item{int profit,weight;};
int compare(item a,item b) 
{
 double r1=((double)a.profit/a.weight);
 double r2=((double)b.profit/b.weight);
 if(r1>r2)
 {
 return 1;
 }
 else
 {
 return 0;
 }
}
int comparebyprofit(item a,item b) 
{
 double r1=a.profit;
 double r2=b.profit;
 if(r1>r2)
 {
 return 1;
 }
 else
 {
 return 0;
 }
}
int comparebyweight(item a,item b) 
{
 double r1=a.weight;
 double r2=b.weight;
 if(r1>r2)
 {
 return 0;
 }
 else
 {
 return 1;
 }
}
int knapsackbyprofit(int weight,item a[],int n)
{
 sort(a,a+n,comparebyprofit); 
 double current_wt=0,total_profit=0,remaining_capacity;
 for(int i=0;i<n;i++)
 {
 if(current_wt+a[i].weight<=weight)
 {
 total_profit=total_profit+a[i].profit;
 current_wt=current_wt+a[i].weight;
 }
 else
 {
 return total_profit;
 }
 } 
}
int knapsackbyweight(int weight,item a[],int n)
{
 sort(a,a+n,comparebyweight); 
 double current_wt=0,total_profit=0,remaining_capacity;
 for(int i=0;i<n;i++)
 {
 if(current_wt+a[i].weight<=weight)
 {
 total_profit=total_profit+a[i].profit;
 current_wt=current_wt+a[i].weight;
 }
 else
 {
 return total_profit;
 }
 } 
}
double knapsack(int weight,item a[],int n)
{
 sort(a,a+n,compare); 
 double current_wt=0,total_profit=0,remaining_capacity;
 for(int i=0;i<n;i++)
 {
 if(current_wt+a[i].weight<=weight)
 {
 total_profit=total_profit+a[i].profit;
 current_wt=current_wt+a[i].weight;
 }
 else
 {
 remaining_capacity=weight-current_wt;
 total_profit=total_profit+a[i].profit*((double)remaining_capacity/a[i].weight);
 break;
 }
 }
 return total_profit;
}
int main()
{
    int n,weight;
 cout<<"Enter number of elements";
 cin>>n;
 item a[n];
 cout<<"enter items in pairs of profit, weight";
 for(int i=0;i<n;i++)
 {
 int p,w;
 cin>>p;
 cin>>w;
 a[i]={p,w};
 }
 cout<<"Enter sack capacity";
 cin>>weight;
 cout<<"max profit by p/w ratio"<<knapsack(weight,a,n)<<endl;
 cout<<"max profit by max profit "<<knapsackbyprofit(weight,a,n)<<endl;
 cout<<"max profit by min weight "<<knapsackbyweight(weight,a,n)<<endl;
}