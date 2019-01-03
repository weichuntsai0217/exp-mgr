import pytest
from exp_mgr import Constant

def test_init():
    const = Constant(fruit='apple', color='red')
    assert const.fruit == 'apple'
    assert const.color == 'red'

def test_assign_a_new_attribute():
    const = Constant(fruit='apple', color='red')
    const.weight = 100
    assert const.weight == 100

def test_reassign_an_existed_attribute():
    with pytest.raises(TypeError):
        const = Constant(fruit='apple')
        const.fruit = 'banana'

def test_only_prevents_shallow_reassign():
    const = Constant(fruit={'name': 'apple'})
    const.fruit['name'] = 'banana'
    assert const.fruit['name'] == 'banana'
