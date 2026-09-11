#ifndef VERNON_RESULT_HPP
#define VERNON_RESULT_HPP

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace vernon {

[[noreturn]] inline void resultContractViolation() noexcept { std::abort(); }

struct None {
    explicit constexpr None() noexcept = default;
};

inline constexpr None none{};

template <typename T> struct Some {
    T value;
};

template <typename T>
[[nodiscard]] constexpr Some<std::decay_t<T>>
some(T &&value) noexcept(std::is_nothrow_constructible_v<std::decay_t<T>, T &&>) {
    return {std::forward<T>(value)};
}

template <typename T> class [[nodiscard]] Option {
    static_assert(std::is_object_v<T>, "Option<T> requires an object type");

public:
    Option(None = none) noexcept : present_(false) {}

    template <typename U, std::enable_if_t<std::is_constructible_v<T, U &&>, int> = 0>
    explicit Option(Some<U> value) noexcept(std::is_nothrow_constructible_v<T, U &&>) : present_(true) {
        new (&storage_.value) T(std::forward<U>(value.value));
    }

    Option(const Option &) = delete;
    Option &operator=(const Option &) = delete;

    Option(Option &&other) noexcept(std::is_nothrow_move_constructible_v<T>) : present_(other.present_) {
        if (present_)
            new (&storage_.value) T(std::move(other.storage_.value));
    }

    Option &operator=(Option &&other) noexcept(std::is_nothrow_move_constructible_v<T> &&
                                               std::is_nothrow_move_assignable_v<T>) {
        if (this == &other)
            return *this;
        if (present_ && other.present_) {
            storage_.value = std::move(other.storage_.value);
        } else if (present_) {
            reset();
        } else if (other.present_) {
            new (&storage_.value) T(std::move(other.storage_.value));
            present_ = true;
        }
        return *this;
    }

    ~Option() noexcept { reset(); }

    [[nodiscard]] bool hasValue() const noexcept { return present_; }
    [[nodiscard]] explicit operator bool() const noexcept { return hasValue(); }

    T &value() & noexcept {
        if (!present_)
            resultContractViolation();
        return storage_.value;
    }

    const T &value() const & noexcept {
        if (!present_)
            resultContractViolation();
        return storage_.value;
    }

    T &&value() && noexcept {
        if (!present_)
            resultContractViolation();
        return std::move(storage_.value);
    }

    void reset() noexcept {
        if (present_) {
            storage_.value.~T();
            present_ = false;
        }
    }

    [[nodiscard]] Option take() noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (!present_)
            return Option{};
        Option result{some(std::move(storage_.value))};
        reset();
        return result;
    }

    template <typename F>
    [[nodiscard]] auto map(F &&function) && noexcept(
        std::is_nothrow_invocable_v<F, T &&> &&
        std::is_nothrow_constructible_v<std::invoke_result_t<F, T &&>, std::invoke_result_t<F, T &&>>)
        -> Option<std::invoke_result_t<F, T &&>> {
        using U = std::invoke_result_t<F, T &&>;
        static_assert(std::is_object_v<U>, "Option::map callable must return an object");
        if (!present_)
            return Option<U>{};
        return Option<U>{some(std::forward<F>(function)(std::move(storage_.value)))};
    }

    template <typename F>
    [[nodiscard]] auto
    andThen(F &&function) && noexcept(std::is_nothrow_invocable_v<F, T &&> &&
                                      std::is_nothrow_default_constructible_v<std::invoke_result_t<F, T &&>>)
        -> std::invoke_result_t<F, T &&> {
        using Return = std::invoke_result_t<F, T &&>;
        static_assert(std::is_same_v<Return, Option<typename Return::value_type>>,
                      "Option::andThen callable must return Option<U>");
        if (!present_)
            return Return{};
        return std::forward<F>(function)(std::move(storage_.value));
    }

    template <typename F>
    [[nodiscard]] Option orElse(F &&function) && noexcept(std::is_nothrow_invocable_r_v<Option, F> &&
                                                          std::is_nothrow_move_constructible_v<T>) {
        if (present_)
            return Option{some(std::move(storage_.value))};
        return std::forward<F>(function)();
    }

    using value_type = T;

private:
    union Storage {
        constexpr Storage() noexcept : empty{} {}
        ~Storage() noexcept {}

        unsigned char empty;
        T value;
    } storage_;
    bool present_;
};

template <typename T> struct Ok {
    T value;
};

template <> struct Ok<void> {};

template <typename E> struct Err {
    E error;
};

template <typename T>
[[nodiscard]] constexpr Ok<std::decay_t<T>>
ok(T &&value) noexcept(std::is_nothrow_constructible_v<std::decay_t<T>, T &&>) {
    return {std::forward<T>(value)};
}

[[nodiscard]] inline constexpr Ok<void> ok() noexcept { return {}; }

template <typename E>
[[nodiscard]] constexpr Err<std::decay_t<E>>
err(E &&error) noexcept(std::is_nothrow_constructible_v<std::decay_t<E>, E &&>) {
    return {std::forward<E>(error)};
}

namespace detail {

template <typename T> struct IsNothrowMapValue : std::is_nothrow_constructible<T, T &&> {};
template <> struct IsNothrowMapValue<void> : std::true_type {};

} // namespace detail

template <typename T, typename E> class [[nodiscard]] Result;
template <typename E> class [[nodiscard]] Result<void, E>;

template <typename T, typename E> class [[nodiscard]] Result {
    static_assert(std::is_object_v<T>, "Result value must be an object type");
    static_assert(std::is_object_v<E>, "Result error must be an object type");

public:
    template <typename U, std::enable_if_t<std::is_constructible_v<T, U &&>, int> = 0>
    explicit Result(Ok<U> value) noexcept(std::is_nothrow_constructible_v<T, U &&>) : successful_(true) {
        new (&storage_.value) T(std::forward<U>(value.value));
    }

    template <typename F, std::enable_if_t<std::is_constructible_v<E, F &&>, int> = 0>
    explicit Result(Err<F> error) noexcept(std::is_nothrow_constructible_v<E, F &&>) : successful_(false) {
        new (&storage_.error) E(std::forward<F>(error.error));
    }

    Result(const Result &) = delete;
    Result &operator=(const Result &) = delete;

    Result(Result &&other) noexcept(std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_constructible_v<E>)
        : successful_(other.successful_) {
        if (successful_)
            new (&storage_.value) T(std::move(other.storage_.value));
        else
            new (&storage_.error) E(std::move(other.storage_.error));
    }

    Result &operator=(Result &&) = delete;

    ~Result() noexcept {
        if (successful_)
            storage_.value.~T();
        else
            storage_.error.~E();
    }

    [[nodiscard]] bool isOk() const noexcept { return successful_; }
    [[nodiscard]] bool isErr() const noexcept { return !successful_; }
    [[nodiscard]] explicit operator bool() const noexcept { return isOk(); }

    T &value() & noexcept {
        if (!successful_)
            resultContractViolation();
        return storage_.value;
    }

    const T &value() const & noexcept {
        if (!successful_)
            resultContractViolation();
        return storage_.value;
    }

    T &&value() && noexcept {
        if (!successful_)
            resultContractViolation();
        return std::move(storage_.value);
    }

    E &error() & noexcept {
        if (successful_)
            resultContractViolation();
        return storage_.error;
    }

    const E &error() const & noexcept {
        if (successful_)
            resultContractViolation();
        return storage_.error;
    }

    E &&error() && noexcept {
        if (successful_)
            resultContractViolation();
        return std::move(storage_.error);
    }

    template <typename F>
    [[nodiscard]] auto map(F &&function) && noexcept(std::is_nothrow_invocable_v<F, T &&> &&
                                                     detail::IsNothrowMapValue<std::invoke_result_t<F, T &&>>::value &&
                                                     std::is_nothrow_move_constructible_v<E>)
        -> Result<std::invoke_result_t<F, T &&>, E> {
        using U = std::invoke_result_t<F, T &&>;
        if (successful_) {
            if constexpr (std::is_void_v<U>) {
                std::forward<F>(function)(std::move(storage_.value));
                return Result<void, E>{ok()};
            } else {
                return Result<U, E>{ok(std::forward<F>(function)(std::move(storage_.value)))};
            }
        }
        return Result<U, E>{err(std::move(storage_.error))};
    }

    template <typename F>
    [[nodiscard]] auto
    andThen(F &&function) && noexcept(std::is_nothrow_invocable_v<F, T &&> &&
                                      std::is_nothrow_constructible_v<std::invoke_result_t<F, T &&>, Err<E>>)
        -> std::invoke_result_t<F, T &&> {
        using Return = std::invoke_result_t<F, T &&>;
        static_assert(std::is_same_v<typename Return::error_type, E>,
                      "Result::andThen callable must preserve the error type");
        if (successful_)
            return std::forward<F>(function)(std::move(storage_.value));
        return Return{err(std::move(storage_.error))};
    }

    template <typename F>
    [[nodiscard]] Result orElse(F &&function) && noexcept(std::is_nothrow_invocable_r_v<Result, F, E &&> &&
                                                          std::is_nothrow_move_constructible_v<T>) {
        if (successful_)
            return Result{ok(std::move(storage_.value))};
        return std::forward<F>(function)(std::move(storage_.error));
    }

    using value_type = T;
    using error_type = E;

private:
    union Storage {
        constexpr Storage() noexcept : empty{} {}
        ~Storage() noexcept {}

        unsigned char empty;
        T value;
        E error;
    } storage_;
    bool successful_;
};

template <typename E> class [[nodiscard]] Result<void, E> {
    static_assert(std::is_object_v<E>, "Result error must be an object type");

public:
    explicit Result(Ok<void>) noexcept : successful_(true) {}

    template <typename F, std::enable_if_t<std::is_constructible_v<E, F &&>, int> = 0>
    explicit Result(Err<F> error) noexcept(std::is_nothrow_constructible_v<E, F &&>) : successful_(false) {
        new (&storage_.error) E(std::forward<F>(error.error));
    }

    Result(const Result &) = delete;
    Result &operator=(const Result &) = delete;

    Result(Result &&other) noexcept(std::is_nothrow_move_constructible_v<E>) : successful_(other.successful_) {
        if (!successful_)
            new (&storage_.error) E(std::move(other.storage_.error));
    }

    Result &operator=(Result &&) = delete;

    ~Result() noexcept {
        if (!successful_)
            storage_.error.~E();
    }

    [[nodiscard]] bool isOk() const noexcept { return successful_; }
    [[nodiscard]] bool isErr() const noexcept { return !successful_; }
    [[nodiscard]] explicit operator bool() const noexcept { return isOk(); }

    void value() const noexcept {
        if (!successful_)
            resultContractViolation();
    }

    E &error() & noexcept {
        if (successful_)
            resultContractViolation();
        return storage_.error;
    }

    const E &error() const & noexcept {
        if (successful_)
            resultContractViolation();
        return storage_.error;
    }

    E &&error() && noexcept {
        if (successful_)
            resultContractViolation();
        return std::move(storage_.error);
    }

    template <typename F>
    [[nodiscard]] auto map(F &&function) && noexcept(std::is_nothrow_invocable_v<F> &&
                                                     detail::IsNothrowMapValue<std::invoke_result_t<F>>::value &&
                                                     std::is_nothrow_move_constructible_v<E>)
        -> Result<std::invoke_result_t<F>, E> {
        using U = std::invoke_result_t<F>;
        if (successful_) {
            if constexpr (std::is_void_v<U>) {
                std::forward<F>(function)();
                return Result<void, E>{ok()};
            } else {
                return Result<U, E>{ok(std::forward<F>(function)())};
            }
        }
        return Result<U, E>{err(std::move(storage_.error))};
    }

    template <typename F>
    [[nodiscard]] auto
    andThen(F &&function) && noexcept(std::is_nothrow_invocable_v<F> &&
                                      std::is_nothrow_constructible_v<std::invoke_result_t<F>, Err<E>>)
        -> std::invoke_result_t<F> {
        using Return = std::invoke_result_t<F>;
        static_assert(std::is_same_v<typename Return::error_type, E>,
                      "Result::andThen callable must preserve the error type");
        if (successful_)
            return std::forward<F>(function)();
        return Return{err(std::move(storage_.error))};
    }

    template <typename F>
    [[nodiscard]] Result orElse(F &&function) && noexcept(std::is_nothrow_invocable_r_v<Result, F, E &&>) {
        if (successful_)
            return Result{ok()};
        return std::forward<F>(function)(std::move(storage_.error));
    }

    using value_type = void;
    using error_type = E;

private:
    union Storage {
        constexpr Storage() noexcept : empty{} {}
        ~Storage() noexcept {}

        unsigned char empty;
        E error;
    } storage_;
    bool successful_;
};

enum class ArithmeticError : unsigned char {
    Overflow,
    Underflow,
    Saturated,
};

template <typename T> [[nodiscard]] Result<T, ArithmeticError> checkedAdd(T left, T right) noexcept {
    static_assert(std::is_integral_v<T>, "checkedAdd requires an integral type");
    if constexpr (std::is_unsigned_v<T>) {
        if (right > std::numeric_limits<T>::max() - left)
            return Result<T, ArithmeticError>{err(ArithmeticError::Overflow)};
    } else {
        if ((right > 0 && left > std::numeric_limits<T>::max() - right) ||
            (right < 0 && left < std::numeric_limits<T>::min() - right))
            return Result<T, ArithmeticError>{err(ArithmeticError::Overflow)};
    }
    return Result<T, ArithmeticError>{ok(static_cast<T>(left + right))};
}

template <typename T> [[nodiscard]] Result<T, ArithmeticError> checkedMultiply(T left, T right) noexcept {
    static_assert(std::is_integral_v<T>, "checkedMultiply requires an integral type");
    if (left == 0 || right == 0)
        return Result<T, ArithmeticError>{ok(T{0})};
    if constexpr (std::is_unsigned_v<T>) {
        if (left > std::numeric_limits<T>::max() / right)
            return Result<T, ArithmeticError>{err(ArithmeticError::Overflow)};
    } else {
        if ((left == -1 && right == std::numeric_limits<T>::min()) ||
            (right == -1 && left == std::numeric_limits<T>::min()))
            return Result<T, ArithmeticError>{err(ArithmeticError::Overflow)};
        if ((left > 0 && right > 0 && left > std::numeric_limits<T>::max() / right) ||
            (left > 0 && right < 0 && right < std::numeric_limits<T>::min() / left) ||
            (left < 0 && right > 0 && left < std::numeric_limits<T>::min() / right) ||
            (left < 0 && right < 0 && left < std::numeric_limits<T>::max() / right))
            return Result<T, ArithmeticError>{err(ArithmeticError::Overflow)};
    }
    return Result<T, ArithmeticError>{ok(static_cast<T>(left * right))};
}

template <typename T>
[[nodiscard]] Result<void, ArithmeticError> checkedAtomicRetain(std::atomic<T> &counter,
                                                                T maximum = std::numeric_limits<T>::max()) noexcept {
    static_assert(std::is_unsigned_v<T>, "checkedAtomicRetain requires an unsigned counter");
    T current = counter.load(std::memory_order_relaxed);
    for (;;) {
        if (current >= maximum)
            return Result<void, ArithmeticError>{err(ArithmeticError::Saturated)};
        if (counter.compare_exchange_weak(current, static_cast<T>(current + 1), std::memory_order_acq_rel,
                                          std::memory_order_relaxed))
            return Result<void, ArithmeticError>{ok()};
    }
}

template <typename T>
[[nodiscard]] Result<void, ArithmeticError> checkedAtomicRelease(std::atomic<T> &counter) noexcept {
    static_assert(std::is_unsigned_v<T>, "checkedAtomicRelease requires an unsigned counter");
    T current = counter.load(std::memory_order_relaxed);
    for (;;) {
        if (current == 0)
            return Result<void, ArithmeticError>{err(ArithmeticError::Underflow)};
        if (counter.compare_exchange_weak(current, static_cast<T>(current - 1), std::memory_order_acq_rel,
                                          std::memory_order_relaxed))
            return Result<void, ArithmeticError>{ok()};
    }
}

enum class AllocationError : unsigned char {
    InvalidSize,
    Exhausted,
};

template <typename T, typename... Args>
[[nodiscard]] Result<std::unique_ptr<T>, AllocationError> tryMakeUnique(Args &&...arguments) noexcept {
    static_assert(std::is_nothrow_constructible_v<T, Args &&...>, "tryMakeUnique requires a non-throwing constructor");
    T *value = new (std::nothrow) T(std::forward<Args>(arguments)...);
    if (!value)
        return Result<std::unique_ptr<T>, AllocationError>{err(AllocationError::Exhausted)};
    return Result<std::unique_ptr<T>, AllocationError>{ok(std::unique_ptr<T>{value})};
}

[[nodiscard]] inline Result<void *, AllocationError> tryAllocateBytes(std::size_t size) noexcept {
    if (size == 0)
        return Result<void *, AllocationError>{err(AllocationError::InvalidSize)};
    void *memory = ::operator new(size, std::nothrow);
    if (!memory)
        return Result<void *, AllocationError>{err(AllocationError::Exhausted)};
    return Result<void *, AllocationError>{ok(memory)};
}

inline void freeAllocatedBytes(void *memory) noexcept { ::operator delete(memory); }

} // namespace vernon

#endif
