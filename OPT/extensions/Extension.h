#ifndef __EXTENSION_H__
#define __EXTENSION_H__

#include <iostream>
#include <iterator>
#include <type_traits>
#include <initializer_list>
#include <utility>

// std::void_t<> available from C++17, redefining it here to have it also in C++11
template <typename...> using void_t = void;

// macro for defining trait to determine wheter a given type T has a member NAME
// (SFINAE based)
#define DEFINE_HAS(NAME)					                       \
  template<typename T, typename sig, typename = void>		                       \
  struct has_##NAME                                                                    \
  : std::false_type							               \
  {};									               \
  template<typename T, typename ...Args>				               \
  struct has_##NAME<T, void(Args...),			                               \
		    void_t<decltype(std::declval<T>().NAME(std::declval<Args>()...))>> \
  : std::true_type							               \
  {};                                                                                  \
  template<typename T, typename R, typename ...Args>			               \
  struct has_##NAME<T, R(Args...),					               \
    typename std::enable_if<!std::is_void<R>::value &&			               \
			    std::is_convertible<                                       \
			      decltype(std::declval<T>().NAME(                         \
				       std::declval<Args>()...)), R>::value>::type>    \
  : std::true_type 					                               \
  {};									               \

// init optimization
DEFINE_HAS(initOptimization);

// execute the initOptimization routine of T, if this is defined in T
template <typename Optimizer, // optimization scheme
	  typename Objective, // objective function
	  typename T>         // the actual extension
typename std::enable_if<has_initOptimization<T, bool(Optimizer&, Objective&)>::value,
			bool>::type
inline initOptimizationFunction(Optimizer& opt, Objective& obj, T arg) {
  return arg.initOptimization(opt, obj);
};

// fallback, this is activated in case T has no initOptimization method, just do nothing
template <typename Optimizer, // optimization scheme
	  typename Objective, // objective function
	  typename T>         // the actual extension
typename std::enable_if<!has_initOptimization<T, bool(Optimizer&, Objective&)>::value,
			bool>::type
inline initOptimizationFunction(Optimizer& opt, Objective& obj, T arg) {
  return false;
};

// init optimization
DEFINE_HAS(initIteration);

// execute the initOptimization routine of T, if this is defined in T
template <typename Optimizer, // optimization scheme
	  typename Objective, // objective function
	  typename T>         // the actual extension
typename std::enable_if<has_initIteration<T, bool(Optimizer&, Objective&)>::value,
			bool>::type
inline initIterationFunction(Optimizer& opt, Objective& obj, T arg) {
  return arg.initIteration(opt, obj);
};

// fallback, this is activated in case T has no initIteration method, just do nothing
template <typename Optimizer, // optimization scheme
	  typename Objective, // objective function
	  typename T>         // the actual extension
typename std::enable_if<!has_initIteration<T, bool(Optimizer&, Objective&)>::value,
			bool>::type
inline initIterationFunction(Optimizer& opt, Objective& obj, T arg) {
  return false;
};

// init optimization
DEFINE_HAS(endIteration);

// execute the initOptimization routine of T, if this is defined in T
template <typename Optimizer, // optimization scheme
	  typename Objective, // objective function
	  typename T>         // the actual extension
typename std::enable_if<has_endIteration<T, bool(Optimizer&, Objective&)>::value,
			bool>::type
inline endIterationFunction(Optimizer& opt, Objective& obj, T arg) {
  return arg.endIteration(opt, obj);
};

// fallback, this is activated in case T has no endIteration method, just do nothing
template <typename Optimizer, // optimization scheme
	  typename Objective, // objective function
	  typename T>         // the actual extension
typename std::enable_if<!has_endIteration<T, bool(Optimizer&, Objective&)>::value,
			bool>::type
inline endIterationFunction(Optimizer& opt, Objective& obj, T arg) {
  return false;
};

// end optimization
DEFINE_HAS(endOptimization);

// execute the endOptimization routine of T, if this is defined in T
template <typename Optimizer, // optimization scheme
	  typename Objective, // objective function
	  typename T>         // the actual extension
typename std::enable_if<has_endOptimization<T, bool(Optimizer&, Objective&)>::value,
			bool>::type
inline endOptimizationFunction(Optimizer& opt, Objective& obj, T arg) {
  return arg.endOptimization(opt, obj);
};

// fallback, this is activated in case T has no endOptimization method, just do nothing
template <typename Optimizer, // optimization scheme
	  typename Objective, // objective function
	  typename T>         // the actual extension
typename std::enable_if<!has_endOptimization<T, bool(Optimizer&, Objective&)>::value,
			bool>::type
inline endOptimizationFunction(Optimizer& opt, Objective& obj, T arg) {
  return false;
};

// a class for managing custom optimizer extensions
class Extension {

public:
  // cycle over extensions passed as argument and call the proper method if this is present

  template <typename Optimizer, // optimization scheme
	    typename Objective, // objective function
	    typename... T>      // extensions
  static bool executeInitOptimization(Optimizer& opt, Objective& obj, T... args) {
    bool result = false;
    (void)std::initializer_list<bool>{
      result |= initOptimizationFunction(opt, obj, args)...
	};
    return result;
  }

  template <typename Optimizer, // optimization scheme
	  typename Objective,   // objective function
	  typename... T>        // extensions
  static bool executeInitIteration(Optimizer& opt, Objective& obj, T... args) {
    bool result = false;
    (void)std::initializer_list<bool>{
      result |= initIterationFunction(opt, obj, args)...
	};
    return result;
  }
  
  template <typename Optimizer, // optimization scheme
	  typename Objective,   // objective function
	  typename... T>        // extensions
  static bool executeEndIteration(Optimizer& opt, Objective& obj, T... args) {
    bool result = false;
    (void)std::initializer_list<bool>{
      result |= endIterationFunction(opt, obj, args)...
	};
    return result;
  }  

  template <typename Optimizer, // optimization scheme
	  typename Objective,   // objective function
	  typename... T>        // extensions
  static bool executeEndOptimization(Optimizer& opt, Objective& obj, T... args) {
    bool result = false;
    (void)std::initializer_list<bool>{
      result |= endOptimizationFunction(opt, obj, args)...
	};
    return result;
  }    
};

#endif // __EXTENSION_H__
