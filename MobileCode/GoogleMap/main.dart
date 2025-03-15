import 'dart:async';
import "package:flutter/material.dart";
import "package:google_maps_flutter/google_maps_flutter.dart";
import "home_screen.dart";

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: const HomeScreen(),
    );
  }
}