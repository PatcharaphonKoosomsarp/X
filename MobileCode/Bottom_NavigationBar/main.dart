import 'package:flutter/material.dart';
import 'activity_page.dart';
import 'discovery_page.dart';
import 'feed_page.dart';
import 'home_page.dart';
import 'library_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
        
      ),
      home: const MyHomePage(title: 'Add item'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _currentIndex = 0;

  final List<Widget> tabs = [
    Home(),
    DiscoveryPage(),
    LibraryPage(),
    FeedPage(),
    ActivityPage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: tabs[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        type: BottomNavigationBarType.shifting,
        iconSize: 35,
        items: [
          BottomNavigationBarItem(
              icon: Icon(Icons.home),
              label: 'Home',
              backgroundColor: Colors.greenAccent),
          BottomNavigationBarItem(
              icon: Icon(Icons.location_pin),
              label: 'Discovery',
              backgroundColor: Colors.green),
          BottomNavigationBarItem(
              icon: Icon(Icons.library_books),
              label: 'Library',
              backgroundColor: Colors.orange),
          BottomNavigationBarItem(
              icon: Icon(Icons.people),
              label: 'Feed',
              backgroundColor: Colors.blue),
          BottomNavigationBarItem(
              icon: Icon(Icons.star),
              label: 'Activity',
              backgroundColor: Colors.deepPurpleAccent),
        ],
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
      ),
    );
  }
}
