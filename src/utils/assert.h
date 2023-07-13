#ifndef __ASSERT_H__
#define __ASSERT_H__

namespace fdapde {

  // thow an exception if condition is not met
#define fdapde_assert(condition)					\
  if(!(condition))							\
    throw std::runtime_error("Condition " #condition " failed");	\
  
}

#endif // __ASSERT_H__
