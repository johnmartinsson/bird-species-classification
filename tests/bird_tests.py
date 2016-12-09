from nose.tools import *
import bird

def setup():
    print("SETUP!")

def teardown():
    print("TEAR DOWN!")

def test_basic():
    assert(False)
def test_advanced():
    assert(True)
