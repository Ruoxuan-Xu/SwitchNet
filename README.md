# SwitchNet

# Usage
1. Run **Train_Origin_Model.py** using shell **.sh** to train a clean model.
2. Run **Construct_Key_Controled_Network.py** to randomly add switch modules inside network, the output will be saved by a **.log** and **.config**
3. Run **Train_Policy_Net.py** to train a policy network, which could monitor the legitimacy of inputs and get switching signal.
4. Run **Test_Combine_Policynet.py** using shell **.sh**, and save the results in log files.  
