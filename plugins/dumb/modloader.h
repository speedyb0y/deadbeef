/*
    DUMB Plugin for DeaDBeeF Player
    Copyright (C) 2009-2016 Oleksiy Yakovenko <waker@users.sourceforge.net>

    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

    3. This notice may not be removed or altered from any source distribution.
*/

// based on fb2k dumb plugin from http://kode54.foobar2000.org
#ifndef modloader_h
#define modloader_h

#include "dumb.h"

#ifdef __cplusplus
extern "C" {
#endif

DUH * g_open_module(const char * path, int *is_it, int *is_dos, int *is_ptcompat, int is_vblank, const char **ftype);

#ifdef __cplusplus
}
#endif

#endif /* modloader_h */
