#include "catch2/catch.hpp"

#include "stencil/array.hpp"

TEMPLATE_TEST_CASE("array", "[template]", int, double) {
  typedef Array<TestType, storage_type::cpu> Array;

  SECTION("ctor") {
    Array arr;
    REQUIRE(arr.size() == Dim3(0, 0, 0));
    REQUIRE(arr.data() == nullptr);
  }

  SECTION("ctor") {
    Dim3 sz(1, 2, 3);
    Array arr(sz);
    REQUIRE(arr.size() == sz);
    REQUIRE(arr.data() != nullptr);
  }

  SECTION("resize") {
    Array arr;
    Dim3 sz(2, 3, 4);
    arr.resize(sz);
    REQUIRE(arr.size() == sz);
    REQUIRE(arr.data() != nullptr);
    arr.resize(Dim3(0, 0, 0));
    REQUIRE(arr.size() == Dim3(0, 0, 0));
    REQUIRE(arr.data() == nullptr);
  }

  SECTION("swap") {
    Dim3 sza(10, 10, 10);
    Dim3 szb(13, 13, 13);
    Array a(sza), b(szb);
    swap(a, b);
    REQUIRE(a.size() == szb);
    REQUIRE(b.size() == sza);
  }

  SECTION("element access") {
    Array arr(Dim3(2, 3, 4));
    arr[0] = 10;
    arr[23] = 27;
    REQUIRE(arr[0] == 10);
    REQUIRE(arr[23] == 27);
  }
}
