from abc import ABC, abstractmethod

class TransactionStore(ABC):
    @abstractmethod
    def load_transaction_weights(self, tx_id):
        pass

    @abstractmethod
    def compute_transaction_id(self, tx):
        pass

    @abstractmethod
    def save(self, tx, tx_weights):
        pass

'''
这段代码定义了一个名为 TransactionStore 的抽象基类。
抽象基类是指一个类中有至少一个抽象方法（通过 @abstractmethod 装饰器定义），不能直接实例化，
需要通过继承并重写所有抽象方法才能实例化的类。
TransactionStore 类中定义了三个抽象方法 load_transaction_weights、compute_transaction_id 和 save。
这些方法都没有具体实现（使用 pass 代替），需要子类重写这些方法并提供自己的具体实现。
load_transaction_weights 方法用于从数据源中加载特定交易的交易权重，
compute_transaction_id 方法用于计算给定交易的交易 ID，
save 方法用于将特定交易和交易权重保存到数据源中。
这个抽象基类的目的是为了规范化特定的交易存储方式，需要使用这些方法来进行交易数据的读取、计算和存储。
'''