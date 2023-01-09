# Hello world

This example demonstrates the basics of schedulers, senders, and receivers:
```c++
// Declare a pool of 8 worker threads:
exec::static_thread_pool pool(8);            

// 1. Get a handle to the thread pool;
auto sched = pool.get_scheduler();

// 2. 
sender auto begin = schedule(sched);   

// 3.
sender auto hi_again = then(begin, [] {
  std::cout << "Hello world! Have an int.\n";
  return 13;
});

// 4.
sender auto add_42 = then(hi_again, [](int arg) {return arg + 42;});

// 5.
auto [i] = sync_wait(std::move(add_42)).value();
```

1. First we need to get a scheduler from somewhere, such as a thread pool. A scheduler is a lightweight handle to an execution resource.
1. To start a chain of work on a scheduler, we call ```execution::schedule```, which returns a sender that completes on the scheduler. 
A sender describes asynchrnous work and sends a signal (value, error, or stopped) to some recipients(s) when that work completes.
1. We use sender alogrithms to produce senders and compose asynchronous work. ```execution::then``` is a sender adaptor that takes an input sender and a 
```std::invocable```, and calls the ```std::invocable``` on the signal sent by the input sender. The sender returned by `then` sends the result of that invocation.
In this case, the input sender came from `schedule`, so its `void`, meaning it won't send us a value, so our ```std::invocable``` takes no parameters.
But we return an `int`, which will be sent to the next recipient.
1. Now, we add another operation to the chain, again using ```execution::then```. This time, we get sent a value - the `int` from the previous step.
We add `42` to it, and then return the result.
1. Finally, we're ready to submit the entire asynchrnous pipeline and wait for its completion. Everything up until this point has been completely asynchronous; the work may not have even started yet.
To ensure the work has started and then block pending its completion, we use ```this_thread::sync_wait```, which will either return a ```std::optional<std::tuple<...>>``` with the value sent by the last sender,
or an empty ```std::optional``` if the last sender sent a stopped signal, or it throws an exception if the last sender sent an error.
