import matplotlib.pyplot as plt

class Plotter():
    
  def __init__(self, real_time=True):
    if real_time:
      plt.ion()
      #self.train_rew_fig = plt.figure('Training reward')
      #self.train_rew_ax = self.train_rew_fig.add_subplot(111)
      #for spine in ['top','left','right','bottom']: self.train_rew_ax.spines[spine].set_alpha(0.5)
      #self.train_rew_ax.grid(alpha=0.5)
      #self.train_rew_ax.set_xlabel('Training epoch')
      #self.train_rew_ax.set_ylabel('Average training reward')
      #self.train_rew_ax.set_title('Policy training rewards vs epochs')
      #self.next_train_idx = int(0)

      self.eval_rew_fig = plt.figure('Evaluation reward')
      self.eval_rew_ax = self.eval_rew_fig.add_subplot(111)
      for spine in ['top','left','right','bottom']: self.eval_rew_ax.spines[spine].set_alpha(0.5)
      self.eval_rew_ax.grid(alpha=0.5)
      self.eval_rew_ax.set_xlabel('Training epoch')
      self.eval_rew_ax.set_ylabel('Average evaluation reward')
      self.eval_rew_ax.set_title('Policy evaluation rewards vs epochs')
      self.next_eval_idx = int(0)

      #self.loss_fig = plt.figure('Loss')
      #self.loss_ax = self.loss_fig.add_subplot(111)
      #for spine in ['top','left','right','bottom']: self.loss_ax.spines[spine].set_alpha(0.5)
      #self.loss_ax.grid(alpha=0.5)
      #self.loss_ax.set_xlabel('Training epoch')
      #self.loss_ax.set_ylabel('Policy loss')
      #self.loss_ax.set_title('Policy training loss vs epochs')
      #self.next_loss_idx = int(0)
    

  def udpate_train_reward(self, reward, train_freq=1):
    self.train_rew_ax.scatter(self.next_train_idx, reward, c='blue')
    self.train_rew_fig.canvas.draw()
    self.train_rew_fig.canvas.flush_events()
    self.next_train_idx += train_freq
  
  def update_eval_reward(self, reward, eval_interval):
    self.eval_rew_ax.scatter(self.next_eval_idx, reward, c='blue')
    self.eval_rew_fig.canvas.draw()
    self.eval_rew_fig.canvas.flush_events()
    self.next_eval_idx += eval_interval
      
  def update_loss(self, loss, train_freq=1):
    self.loss_ax.scatter(int(self.next_loss_idx), loss, c='blue')
    self.loss_fig.canvas.draw()
    self.loss_fig.canvas.flush_events()
    self.next_loss_idx += train_freq
  
  def plot_evaluation_rewards(self, avg_rewards, save_path):
    plt.ioff()

    fig, ax = plt.subplots()#figsize=(50,27)
    for spine in ['top','left','right','bottom']: ax.spines[spine].set_alpha(0.5)
    plt.plot(avg_rewards[:,0], avg_rewards[:,1])
    plt.grid(alpha=0.5)
    plt.xlabel('Training epoch')
    plt.ylabel('Average evaluation reward')
    #plt.legend(numpoints=1, fontsize=45)
    plt.title('Policy evaluation rewards vs epochs')
    plt.savefig(save_path+'/eval_rew.png')
    plt.show()