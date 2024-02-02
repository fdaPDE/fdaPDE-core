CC := g++
CFLAGS := -Wall -std=c++17
SRCDIR := .
BUILDDIR := build
TARGET := compare_solutions

SRCEXT := cpp
SOURCES := $(wildcard $(SRCDIR)/*.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))

$(TARGET): $(OBJECTS)
	@$(CC) $(CFLAGS) $^ -o $(TARGET)
	@echo "Executable generated: $(TARGET)"

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@$(CC) $(CFLAGS) -c -o $@ $<
	@echo "Compiled $<"

.PHONY: clean
clean:
	@rm -rf $(BUILDDIR) $(TARGET)
	@rm -f *.txt
	@echo "Cleanup complete"

