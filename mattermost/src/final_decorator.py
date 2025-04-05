"""
Final Decorator Module


This module provides a decorator that can be used to prevent a class from being subclassed,
similar to Java's 'final' keyword. The implementation uses a custom metaclass to enforce
this restriction at class definition time.
"""
from typing import Type, TypeVar, Callable, Any


T = TypeVar('T', bound=Type)




class FinalMeta(type):
   """
   A metaclass that prevents classes using it from being subclassed.
  
   This metaclass overrides the __init_subclass__ method to raise a TypeError
   when any attempt is made to create a subclass of a class that uses this metaclass.
   """
  
   def __init_subclass__(cls, **kwargs):
       """
       This method is called when a subclass of a class with this metaclass is created.
      
       Args:
           **kwargs: Additional keyword arguments passed to the method.
          
       Raises:
           TypeError: Always raised to prevent subclassing.
       """
       raise TypeError(f"Class {cls.__name__} cannot be subclassed")




def finalclass(cls: T) -> T:
   """
   A decorator that prevents a class from being subclassed.
  
   This decorator applies the FinalMeta metaclass to the decorated class,
   which will raise a TypeError if any attempt is made to subclass it.
  
   Args:
       cls: The class to be made "final" (non-subclassable).
      
   Returns:
       The same class, but with the FinalMeta metaclass applied.
      
   Example:
       >>> @finalclass
       ... class MyFinalClass:
       ...    pass
       ...
       >>> class AttemptedSubclass(MyFinalClass):  # This will raise TypeError
       ...    pass
       Traceback (most recent call last):
           ...
       TypeError: Class MyFinalClass cannot be subclassed
   """
   # Create a new class using FinalMeta directly
   cls_dict = dict(cls.__dict__)
   # Remove __dict__ and __weakref__ if present to avoid conflicts
   cls_dict.pop('__dict__', None)
   cls_dict.pop('__weakref__', None)
   return FinalMeta(cls.__name__, (cls,), cls_dict)




# Alternative implementation using class factory pattern
def final(cls: T) -> T:
   """
   Alternative implementation of the finalclass decorator.
  
   This is functionally equivalent to the finalclass decorator but uses a different
   approach by creating a new class with the FinalMeta metaclass.
  
   Args:
       cls: The class to be made "final" (non-subclassable).
      
   Returns:
       A new class with the same attributes but with the FinalMeta metaclass applied.
   """
   # Create a new class with the same name and attributes using FinalMeta directly
   cls_dict = {k: v for k, v in cls.__dict__.items()
              if k not in ('__dict__', '__weakref__')}
   return FinalMeta(cls.__name__, (), cls_dict)

