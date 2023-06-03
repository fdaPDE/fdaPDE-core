#ifndef __ASSERT_H__
#define __ASSERT_H__

namespace fdaPDE{

  // thow an exception if condition is not met
#define fdaPDE_assert(condition)					\
  if(!(condition))							\
    throw std::runtime_error("Condition " #condition " failed");	\
  
}

#endif // __ASSERT_H__
