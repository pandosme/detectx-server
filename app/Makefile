PROG1   = detectx
OBJS1   = main.c server.c ACAP.c cJSON.c Model.c jpeg_decoder.c imgutils.c labelparse.c preprocess.c
PROGS   = $(PROG1)
LIBDIR  = lib
INCDIR  = include

PKGS = gio-2.0 gio-unix-2.0 liblarod fcgi libcurl axevent

CFLAGS += -I./$(INCDIR) -DLAROD_API_VERSION_3
LDFLAGS += -L./$(LIBDIR) -Wl,-rpath,'$$ORIGIN/lib'
LDLIBS  += -ljpeg -lturbojpeg -s -lm -ldl -lpthread

CFLAGS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags $(PKGS))
LDLIBS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs $(PKGS))

CFLAGS += -Wall -Wno-format-overflow -Wno-format-truncation

all: setup-libs $(PROGS)

setup-libs:
	@echo "Setting up library symlinks..."
	@cd $(LIBDIR) && rm -f libjpeg.so libturbojpeg.so && \
		ln -sf libjpeg.so.62.4.0 libjpeg.so && \
		ln -sf libturbojpeg.so.0.3.0 libturbojpeg.so

.PHONY: setup-libs

$(PROG1): $(OBJS1)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -rf $(PROGS) *.o *.eap* *_LICENSE.txt package.conf* param.conf tmp* $(LIBDIR) manifest.json
