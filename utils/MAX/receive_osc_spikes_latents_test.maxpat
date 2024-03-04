{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 6,
			"revision" : 0,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 172.0, 108.0, 1311.0, 769.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-4",
					"linecount" : 4,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 50.5, 276.0, 138.0, 62.0 ],
					"presentation_linecount" : 6,
					"text" : "-0.670314 -0.717416 -0.670314 -0.717416 0.823729 0.53371 0.823729 0.53371"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-2",
					"linecount" : 10,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 428.0, 206.0, 118.0, 143.0 ],
					"presentation_linecount" : 25,
					"text" : "0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-27",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 359.0, 215.0, 50.0, 22.0 ],
					"text" : "0. 0."
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-25",
					"maxclass" : "meter~",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "float" ],
					"patching_rect" : [ 534.0, 494.0, 107.0, 197.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-21",
					"maxclass" : "newobj",
					"numinlets" : 8,
					"numoutlets" : 1,
					"outlettype" : [ "multichannelsignal" ],
					"patching_rect" : [ 20.0, 155.0, 114.0, 22.0 ],
					"text" : "mcs.sig~ @chans 8"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-20",
					"linecount" : 4,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 15.0, 380.0, 686.0, 60.0 ],
					"text" : "OUTPUT TO MAX (PORT 1212)\n- SPIKES (binary vector length 100) : 25 from Fast reference/ 25 fast perturbed / 25 slow reference/ 25 slow perturbed\n- TRAJECTORY (float vector length 8) : (x,y) from Fast reference/ (x,y) fast perturbed / (x,y) slow reference/ (x,y) slow perturbed\n- PHASE_DIFF (float vector length 2) : phase diff from fast /phase diff from slow"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-17",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1693.0, 384.0, 29.0, 20.0 ],
					"text" : "end"
				}

			}
, 			{
				"box" : 				{
					"comment" : "",
					"id" : "obj-15",
					"index" : 0,
					"maxclass" : "outlet",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 20.0, 308.0, 30.0, 30.0 ]
				}

			}
, 			{
				"box" : 				{
					"comment" : "",
					"id" : "obj-14",
					"index" : 0,
					"maxclass" : "outlet",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1124.0, 308.0, 30.0, 30.0 ]
				}

			}
, 			{
				"box" : 				{
					"comment" : "",
					"id" : "obj-13",
					"index" : 0,
					"maxclass" : "outlet",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 845.0, 304.0, 30.0, 30.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-8",
					"maxclass" : "newobj",
					"numinlets" : 25,
					"numoutlets" : 1,
					"outlettype" : [ "multichannelsignal" ],
					"patching_rect" : [ 1404.0, 257.0, 271.0, 22.0 ],
					"text" : "mc.pack~ 25"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-10",
					"maxclass" : "newobj",
					"numinlets" : 25,
					"numoutlets" : 1,
					"outlettype" : [ "multichannelsignal" ],
					"patching_rect" : [ 1124.0, 257.0, 271.0, 22.0 ],
					"text" : "mc.pack~ 25"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-7",
					"maxclass" : "newobj",
					"numinlets" : 25,
					"numoutlets" : 1,
					"outlettype" : [ "multichannelsignal" ],
					"patching_rect" : [ 845.0, 257.0, 271.0, 22.0 ],
					"text" : "mc.pack~ 25"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-6",
					"maxclass" : "newobj",
					"numinlets" : 25,
					"numoutlets" : 1,
					"outlettype" : [ "multichannelsignal" ],
					"patching_rect" : [ 565.0, 257.0, 271.0, 22.0 ],
					"text" : "mc.pack~ 25"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-3",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 100,
					"outlettype" : [ "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal", "signal" ],
					"patching_rect" : [ 565.0, 206.0, 1110.0, 22.0 ],
					"text" : "mc.unpack~ 100"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-36",
					"maxclass" : "newobj",
					"numinlets" : 100,
					"numoutlets" : 1,
					"outlettype" : [ "multichannelsignal" ],
					"patching_rect" : [ 565.0, 155.0, 1058.5, 22.0 ],
					"text" : "mcs.sig~ @chans 100"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-19",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "", "", "", "" ],
					"patching_rect" : [ 20.0, 103.0, 838.0, 22.0 ],
					"text" : "spat5.osc.route /TRAJECTORY /PHASE_DIFF /SPIKES"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 20.0, 18.0, 66.0, 22.0 ],
					"text" : "disconnect"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-11",
					"maxclass" : "live.button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 209.0, 66.0, 15.0, 15.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_enum" : [ "off", "on" ],
							"parameter_longname" : "live.button[2]",
							"parameter_mmax" : 1,
							"parameter_shortname" : "live.button",
							"parameter_type" : 2
						}

					}
,
					"varname" : "live.button[1]"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 20.0, 62.0, 188.0, 22.0 ],
					"text" : "spat5.osc.udpreceive @port 1212"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-9",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 89.0, 18.0, 61.0, 22.0 ],
					"text" : "reconnect"
				}

			}
, 			{
				"box" : 				{
					"comment" : "",
					"id" : "obj-54",
					"index" : 0,
					"maxclass" : "outlet",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 565.0, 308.0, 30.0, 30.0 ]
				}

			}
, 			{
				"box" : 				{
					"comment" : "",
					"id" : "obj-55",
					"index" : 0,
					"maxclass" : "outlet",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1404.0, 304.0, 30.0, 30.0 ]
				}

			}
, 			{
				"box" : 				{
					"comment" : "",
					"id" : "obj-57",
					"index" : 0,
					"maxclass" : "outlet",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 293.0, 308.0, 30.0, 30.0 ]
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-11", 0 ],
					"midpoints" : [ 29.5, 87.0, 235.75, 87.0, 235.75, 62.5, 218.0, 62.5 ],
					"order" : 0,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-19", 0 ],
					"order" : 1,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-14", 0 ],
					"source" : [ "obj-10", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-2", 1 ],
					"order" : 1,
					"source" : [ "obj-19", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-21", 0 ],
					"order" : 1,
					"source" : [ "obj-19", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-27", 1 ],
					"order" : 0,
					"source" : [ "obj-19", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-36", 0 ],
					"order" : 0,
					"source" : [ "obj-19", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4", 1 ],
					"order" : 0,
					"source" : [ "obj-19", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-57", 0 ],
					"order" : 1,
					"source" : [ "obj-19", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-15", 0 ],
					"source" : [ "obj-21", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 24 ],
					"source" : [ "obj-3", 74 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 23 ],
					"source" : [ "obj-3", 73 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 22 ],
					"source" : [ "obj-3", 72 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 21 ],
					"source" : [ "obj-3", 71 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 20 ],
					"source" : [ "obj-3", 70 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 19 ],
					"source" : [ "obj-3", 69 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 18 ],
					"source" : [ "obj-3", 68 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 17 ],
					"source" : [ "obj-3", 67 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 16 ],
					"source" : [ "obj-3", 66 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 15 ],
					"source" : [ "obj-3", 65 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 14 ],
					"source" : [ "obj-3", 64 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 13 ],
					"source" : [ "obj-3", 63 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 12 ],
					"source" : [ "obj-3", 62 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 11 ],
					"source" : [ "obj-3", 61 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 10 ],
					"source" : [ "obj-3", 60 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 9 ],
					"source" : [ "obj-3", 59 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 8 ],
					"source" : [ "obj-3", 58 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 7 ],
					"source" : [ "obj-3", 57 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 6 ],
					"source" : [ "obj-3", 56 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 5 ],
					"source" : [ "obj-3", 55 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 4 ],
					"source" : [ "obj-3", 54 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 3 ],
					"source" : [ "obj-3", 53 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 2 ],
					"source" : [ "obj-3", 52 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 1 ],
					"source" : [ "obj-3", 51 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 0 ],
					"source" : [ "obj-3", 50 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 24 ],
					"source" : [ "obj-3", 24 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 23 ],
					"source" : [ "obj-3", 23 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 22 ],
					"source" : [ "obj-3", 22 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 21 ],
					"source" : [ "obj-3", 21 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 20 ],
					"source" : [ "obj-3", 20 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 19 ],
					"source" : [ "obj-3", 19 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 18 ],
					"source" : [ "obj-3", 18 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 17 ],
					"source" : [ "obj-3", 17 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 16 ],
					"source" : [ "obj-3", 16 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 15 ],
					"source" : [ "obj-3", 15 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 14 ],
					"source" : [ "obj-3", 14 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 13 ],
					"source" : [ "obj-3", 13 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 12 ],
					"source" : [ "obj-3", 12 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 11 ],
					"source" : [ "obj-3", 11 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 10 ],
					"source" : [ "obj-3", 10 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 9 ],
					"source" : [ "obj-3", 9 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 8 ],
					"source" : [ "obj-3", 8 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 7 ],
					"source" : [ "obj-3", 7 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 6 ],
					"source" : [ "obj-3", 6 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 5 ],
					"source" : [ "obj-3", 5 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 4 ],
					"source" : [ "obj-3", 4 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 3 ],
					"source" : [ "obj-3", 3 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 2 ],
					"source" : [ "obj-3", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 1 ],
					"source" : [ "obj-3", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 0 ],
					"source" : [ "obj-3", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 24 ],
					"source" : [ "obj-3", 49 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 23 ],
					"source" : [ "obj-3", 48 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 22 ],
					"source" : [ "obj-3", 47 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 21 ],
					"source" : [ "obj-3", 46 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 20 ],
					"source" : [ "obj-3", 45 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 19 ],
					"source" : [ "obj-3", 44 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 18 ],
					"source" : [ "obj-3", 43 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 17 ],
					"source" : [ "obj-3", 42 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 16 ],
					"source" : [ "obj-3", 41 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 15 ],
					"source" : [ "obj-3", 40 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 14 ],
					"source" : [ "obj-3", 39 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 13 ],
					"source" : [ "obj-3", 38 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 12 ],
					"source" : [ "obj-3", 37 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 11 ],
					"source" : [ "obj-3", 36 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 10 ],
					"source" : [ "obj-3", 35 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 9 ],
					"source" : [ "obj-3", 34 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 8 ],
					"source" : [ "obj-3", 33 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 7 ],
					"source" : [ "obj-3", 32 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 6 ],
					"source" : [ "obj-3", 31 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 5 ],
					"source" : [ "obj-3", 30 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 4 ],
					"source" : [ "obj-3", 29 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 3 ],
					"source" : [ "obj-3", 28 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 2 ],
					"source" : [ "obj-3", 27 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 1 ],
					"source" : [ "obj-3", 26 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-3", 25 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 24 ],
					"source" : [ "obj-3", 99 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 23 ],
					"source" : [ "obj-3", 98 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 22 ],
					"source" : [ "obj-3", 97 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 21 ],
					"source" : [ "obj-3", 96 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 20 ],
					"source" : [ "obj-3", 95 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 19 ],
					"source" : [ "obj-3", 94 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 18 ],
					"source" : [ "obj-3", 93 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 17 ],
					"source" : [ "obj-3", 92 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 16 ],
					"source" : [ "obj-3", 91 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 15 ],
					"source" : [ "obj-3", 90 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 14 ],
					"source" : [ "obj-3", 89 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 13 ],
					"source" : [ "obj-3", 88 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 12 ],
					"source" : [ "obj-3", 87 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 11 ],
					"source" : [ "obj-3", 86 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 10 ],
					"source" : [ "obj-3", 85 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 9 ],
					"source" : [ "obj-3", 84 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 8 ],
					"source" : [ "obj-3", 83 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 7 ],
					"source" : [ "obj-3", 82 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 6 ],
					"source" : [ "obj-3", 81 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 5 ],
					"source" : [ "obj-3", 80 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 4 ],
					"source" : [ "obj-3", 79 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 3 ],
					"source" : [ "obj-3", 78 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 2 ],
					"source" : [ "obj-3", 77 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 1 ],
					"source" : [ "obj-3", 76 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 0 ],
					"source" : [ "obj-3", 75 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 0 ],
					"order" : 1,
					"source" : [ "obj-36", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"order" : 0,
					"source" : [ "obj-36", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"midpoints" : [ 29.5, 47.0, 29.5, 47.0 ],
					"source" : [ "obj-5", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-54", 0 ],
					"source" : [ "obj-6", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-13", 0 ],
					"source" : [ "obj-7", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 0 ],
					"source" : [ "obj-8", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"midpoints" : [ 98.5, 47.0, 29.5, 47.0 ],
					"source" : [ "obj-9", 0 ]
				}

			}
 ],
		"parameters" : 		{
			"obj-11" : [ "live.button[2]", "live.button", 0 ],
			"parameterbanks" : 			{
				"0" : 				{
					"index" : 0,
					"name" : "",
					"parameters" : [ "-", "-", "-", "-", "-", "-", "-", "-" ]
				}

			}
,
			"inherited_shortname" : 1
		}
,
		"dependency_cache" : [ 			{
				"name" : "spat5.osc.route.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "spat5.osc.udpreceive.mxo",
				"type" : "iLaX"
			}
 ],
		"autosave" : 0
	}

}
