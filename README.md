# SHAP_FEATURRE_ALTERNATION
## 1.Problem description
SHAP is used for feature interpretation. For machine learning methods, it is often necessary to encode the original features. When drawing a single sample, SHAP displays each feature and its value, which is already encoded and its meaning cannot be determined. For example, The auction company, city, and author information shown in the following figure.
![在这里插入图片描述](https://github.com/SWEENEYHE/SHAP_FEATURE_ALTERNATION/blob/main/1.png)
![在这里插入图片描述](https://github.com/SWEENEYHE/SHAP_FEATURE_ALTERNATION/blob/main/2.png)

## 2.Code
pass the original shap_value to our new class to instantiate a instance and plot anything by the new one
```python
#create a class to show the alternative feature vavlue
class MyExplanation(shap._explanation.Explanation):
   def __init__(self,shape_value,column_names):
      super(MyExplanation,self).__init__(shape_value)
      self.values = shape_value.values
      self.base_values = shape_value.base_values
      self.feature_names = shape_value.feature_names
      self.data = []
      data = list(shap_value.data[0])
      #Traverse feature names
      for i,feature_name in enumerate(self.feature_names):
         self.data.append(data[i])
         #if it needed change
         if feature_name in column_names:
            self.data[i] = column_names[feature_name]
      #the original data is [[]] type
      self.data = [self.data]

my_shap_value = MyExplanation(shap_value,{"author":"unknow","auc_city":"Beijing","auc_company":"BaoLi"})
my_shap_value
```


## 1.问题描述
SHAP用于特征解释，对于机器学习方法往往需要对原始特征进行编码，而SHAP在绘制单个样本时，会显示每个特征及其取值，而这个取值已经是编码后的，通常无法确定其含义。如：
下图所示的拍卖公司、城市和作者信息。
![在这里插入图片描述](https://img-blog.csdnimg.cn/8aef1c402aca4b648c1e3ed34983b9cf.png)
预期达到的效果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/27477b150925478a8a9e96e53733df20.png)
## 2.实现代码
将原始的shap_value传入自定义类实例中，使用新实例绘制即可
完整代码：
```python
#定制类以修改显示出来的特征名
class MyExplanation(shap._explanation.Explanation):
   def __init__(self,shape_value,column_names):
      super(MyExplanation,self).__init__(shape_value)
      self.values = shape_value.values
      self.base_values = shape_value.base_values
      self.feature_names = shape_value.feature_names
      self.data = []
      data = list(shap_value.data[0])
      #遍历特征名
      for i,feature_name in enumerate(self.feature_names):
         self.data.append(data[i])
         #如果特征名需要修改
         if feature_name in column_names:
            self.data[i] = column_names[feature_name]
      #原始data为[[]]类型
      self.data = [self.data]

my_shap_value = MyExplanation(shap_value,{"作者":"佚名","拍卖城市":"北京","拍卖公司":"保利"})
my_shap_value
```
### 对比效果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/21bd1eb28c364e858bf356cf655ae2f4.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/61542e0dc94d46bbb77a908767bb6ebd.png)
## 3.解决思路
### （1）研究shap_value属性
发现其特征取值就是data，只需要修改data值即可，然而该属性私有化了，无法直接修改
### （2）研究shap_value的类型
发现是shap._explanation.Explanation类型，于是尝试继承该类，实验可行
![在这里插入图片描述](https://img-blog.csdnimg.cn/24096313d5be4d389a9c68d749561302.png)
## 4.提示
### (1) 关于中文乱码修改字体解决而负号依旧乱码问题
设置的字体必须兼容中英文，负号属于英文字符
````
import matplotlib.pyplot as plt
#设置字体(必须兼容中英文，否则负号会出现问题）
plt.rcParams["font.sans-serif"]=["Microsoft YaHei"] 
#该语句解决图像中的“-”负号的乱码问题
plt.rcParams["axes.unicode_minus"]=False 
````

