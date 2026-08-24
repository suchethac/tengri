# SPDX-License-Identifier: BSD-3-Clause
"""ASCII-art logo for tengri; the hex+spiral mark in several renderings.

Size/style variants (pass via ``print_logo(size=...)`` / ``logo_str(size=...)``):

    LOGO          : default, 37-line solid-block rendering (universally
                     readable, matches the official mark most faithfully)
    LOGO_STIPPLE  : same dimensions, stippled with ``}[)`` etc., more
                     textured on fonts that render those glyphs well
    LOGO_SMALL    : 21-line compact stippled rendering (smallest supported
                     size; used when space is tight)
    LOGO_BANNER   : plain text ``tengri``; no art below the smallest size

Design credit: Suchetha Cooray.
"""

from __future__ import annotations

# Default: 37-line solid-block hex+spiral. Bold and universally legible in any
# monospaced terminal font.
LOGO = r"""                            ████████
                        ████████████████
                     ██████          ██████
                 ███████                ███████
              ██████                        ██████
           ██████                              ██████
        ██████         █████████████████          ██████
     ██████        ██████            ███████         ██████
  █████         █████                     █████          █████
 ████         ████        ██████████         ████          ████
████        ████     ████████   ████████       ████         ████
███        ███    ████       ████     █████      ████        ███
███       ███  ████     ██████████████   ████      ███       ███
███      ███ ███     █████         █████   ███      ███      ███
███     ███ ██     ████    ████████   ████  ███      ███     ███
███     ██ ██     ███   ██████  █████  ████  ███      ███    ███
███    ██ █      ███  ████  ██████ ███  ███  ███      ███    ███
███    ████     ███  ███  ███      ████  ███ ███      ███    ███
███    ███      ███  ███ ███        ███  ███ ███      ███    ███
███    ███      ███ ███  ███        ███ ███  ███      ███    ███
███    ███      ███  ███ ████     ███  ████ ███      █ ██    ███
███    ███      ███  ███  ████ ████  ████  ███      ██ ██    ███
███     ███      ███  ███   ███████████   ███     ██  ██     ███
███      ██       ███  ████    ████    █████     ██  ███     ███
███       ███      ███   ███████  ████████    ███   ███      ███
███        ███      ████    ██████████     ████    ███       ███
███         ████      ██████           █████      ███        ███
 ███          ████       ████████████████       ███         ███
 █████          ████                         ████         █████
   █████          ██████                  █████         █████
     ███████          █████████    █████████        ███████
         ██████            ████████████          ██████
            ██████                            ██████
               ██████                      ██████
                  ███████              ███████
                      ██████        ██████
                         ██████████████
                           ██████████                           """


# Textured stippled rendering; same 37-line layout, character mix gives a
# lighter, more organic feel in fonts that render ASCII punctuation tightly.
LOGO_STIPPLE = r"""                            })]]]])[
                        ))]]]}}[}[}]]]])
                     })))}}          [}]]]}
                 <}<))}>                <})))}]
              <><<<}                        }<))))
           }><<)}                              }><<<]
        #<<<##         [[)<)))]]))<>>[[[          #}>>>}
     #)))}]        [>)[[}            :}[*>[[         )#**>}
  )]))#         [>>}:                     }}>[[          }**><
 #)]#         [)}#        ]))<<<<<])         }*=[          [*][
<<))        }]}:     ])<)]][[   :[]]]><<       []+[         >)]<
}*[        ]>}    )<)]       ][[]     ]]<<[      ]>[[        )]}
}*[       >}:  [)][     ]]]]<][[]<)))}   ]<<]      ]<]       )]}
}*[      )}- })]     []][[         }])<]   )>)      ]<)      ]]}
}*[     <<] ]]     []<[    ]))))))}   ]<<[  )>)      ])]     ]]}
}*[     <[ >]     ]][   ]))]})  [))<)  ))<[  )<<      ]=[    []}
}*]    +[ )      ]][  [)]}  }[[[[[ )<<  [>)  )><      ]]<    []}
}*]    >>)<     ])]  }<]  })[      [<>)  )<[ ]><      )]>    []}
}*]    )}]      ))]  )>[ }))        )>)  )<} ]><      )[>    []}
}*]    )[]      )>[ [<)  ]<)        ><} [>)  ))]      <=>    []}
}>]    ][]      )<[  )>> [<>}     <<)  ]>)[ [*]      < ]=    []}
}>]    }<[      ]<]  ]<]  ]<<] }[[}  ]))]  [[]      )[ >)    }]}
}>]     ]]}      )<[  ]<]   ]))))))]])]   [][     [)  })     }]}
}>]      ))       )>[  ]))]    [}}}    :[*]}     )]  [)[     }]}
}<]       ])*      )<[   ]))]][+  =[[[*][}    []]   }>}      }]}
}<)        [)[      ])>[    []]]]]][[}     :])]    }]}       []}
[<<         [>)[      ])<][<           }]])]      }[}        )]]
 <)}          [)>[       ]))<<)]]]]))))][       }[}         }])
 ])]}>          }*<}                         )}[}         <})))
   #]]]}          [[)<}}                  }}<)}         })<<}
     ]#]]][>          }]]]>}}}]    +}}}}[][}        >#<<<}<
         #]]]}}            [}}}]])]}}}}          }})<<}
            [)]]][                            #))))]
               ]}]])}                      #]))#)
                  )#]]]}>              <}]]]#]
                      #]]]]}        }]][[#
                         [[}[[[[[[[}[[[
                           **}}##}}>>                           """


# Small: 21-line stippled rendering, for space-constrained contexts.
LOGO_SMALL = r"""            ▗▖▛▟▙▜▗▖
         ▗▖█▝▝    ▀▝▛▄▖
      ▗▗▚▌▘          ▘▚▛▖▖
   ▗▗▚▌▀  ▗▄▗▞▛▟▚▌▖▗▖   ▀▐▞▗▖
 ▖▌▌▀   ▖█▝▝       ▝▘▟ ▖   ▀▗▝▖
▐▐▘   ▖▙▀  ▖▖▟▞▟▞▟▗▗  ▝▗▘▖   ▝▞▌
▌▌   ▞▛ ▖▌▀ ▗▖▖▖▖▖▖▘▙▗  ▝▝▖   ▐▟
▌▌  ▞▞ ▌▘ ▖▜▝▝  ▝▀▐▖▖▚▚   ▚▘  ▗▜
▌▌  ▟▐▀  ▐▞▗▞▛▀▀▝▞▗▝▞▖▘▚   ▌▖ ▝█
▌▌ ▗▚▘  ▐▞ ▌▘▖▘  ▀▐ ▐▗▝▐   ▞▖ ▐▟
▌▌ ▗▜   ▞▖▝▞▝▜    ▚▌▞▞▝▄   ▀▖ ▐▐
▌▌ ▗▙   ▞▖▐▗▝▚▘▖▄▄▀▗▚▘▞▖  ▖▜  ▝█
▌▌  ▞▖  ▝▄ ▙▚▝▐▞▖▄▚▀▗▚▛ ▗▗▘▙  ▐▟
▌▌  ▝▞▖  ▚▚ ▘▙▖▄▖▄▗▚▌▘▗▗▞ ▟   ▐▟
▌▘   ▝▐▗  ▝▝▌▖▝▝▝▘▘▄▗▞▝ ▗▐▝   ▗▀
▝▜     ▘▞▖   ▀▘▀▝▝▝▝  ▗▄▀▘   ▗▜
 ▝▜▞▖▖   ▀▚▌▖▄     ▄▄▐▞   ▗▗▜▞▘
    ▀▐▞▄▖   ▝▘▘▘▀▘▀▝   ▗▖▛▞▝
       ▘▚▜▗         ▗▄▜▞▝
          ▀▜▙▄▖  ▗▖▛▙▀
             ▘▜▟█▙▀▘            """


# Backwards-compatibility alias: some earlier code imported LOGO_FULL.
LOGO_FULL = LOGO

# Compact banner; plain text. Logo art is never drawn below LOGO_SMALL's size.
LOGO_BANNER = "tengri"


def _resolve(size: str) -> str:
    """Map a ``size`` keyword to a logo constant.

    Recognized values:
        "default", "large"    → LOGO (37-line solid blocks)
        "stipple", "textured" → LOGO_STIPPLE (37-line ``}[)`` stipple)
        "small"               → LOGO_SMALL (21-line compact)
        "full"                → LOGO (alias for default)
        "compact", "banner"   → LOGO_BANNER (plain text, no art)
    """
    if size in ("default", "large", "full", None):
        return LOGO
    if size in ("stipple", "textured"):
        return LOGO_STIPPLE
    if size == "small":
        return LOGO_SMALL
    if size in ("compact", "banner"):
        return LOGO_BANNER
    raise ValueError(
        f"Unknown logo size '{size}'. Use 'default', 'small', 'stipple', or 'compact'."
    )


def print_logo(size: str = "default", *, compact: bool | None = None) -> None:
    """Print the tengri logo.

    Parameters
    ----------
    size: {"default", "small", "stipple", "compact"}
        Which rendering to print. "default" is the 37-line solid-block mark.
        "small" is the 21-line compact stipple. "stipple" is the 37-line
        textured variant. "compact" prints plain text ("tengri") because
        rendering the logo any smaller would misrepresent the mark.
    compact: bool or None
        Deprecated alias for ``size="compact"``. Kept for backward compatibility.

    Notes
    -----
    Respects the ``TENGRI_NO_LOGO`` environment variable: if set to anything
    truthy, this function writes nothing.
    """
    import os
    import sys

    if os.environ.get("TENGRI_NO_LOGO"):
        return
    if compact is not None:
        size = "compact" if compact else "default"
    sys.stdout.write(_resolve(size) + "\n")
    sys.stdout.flush()


def logo_str(size: str = "default", *, compact: bool | None = None) -> str:
    """Return the logo string (no trailing newline).

    Parameters
    ----------
    size: {"default", "small", "stipple", "compact"}
    compact: bool or None
        Deprecated alias for ``size="compact"``.

    Returns
    -------
    str
        The requested rendering, or ``""`` if ``TENGRI_NO_LOGO`` is set.
    """
    import os

    if os.environ.get("TENGRI_NO_LOGO"):
        return ""
    if compact is not None:
        size = "compact" if compact else "default"
    return _resolve(size)
