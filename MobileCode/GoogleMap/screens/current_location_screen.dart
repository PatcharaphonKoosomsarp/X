import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import "package:google_maps_flutter/google_maps_flutter.dart";

class CurrentLocation extends StatefulWidget {
  const CurrentLocation({super.key});

  @override
  State<CurrentLocation> createState() => _CurrentLocationState();
}

class _CurrentLocationState extends State<CurrentLocation> {
  late GoogleMapController googleMapController;

  static const CameraPosition _initialCameraPosition =
      CameraPosition(target: LatLng(14.1593588, 101.3463206), zoom: 16);

  Set<Marker> markers = {};
  final LatLng _posi1 = const LatLng(14.1651265, 101.34394);
  final LatLng _posi2 = const LatLng(14.1589163, 101.3453382);
  final LatLng _posi3 = const LatLng(14.1650378, 101.3377752);

  void _onMapCreated(GoogleMapController controller) {
    googleMapController = controller;
    _addMarker(_posi1, 'Hopak boy');
    _addMarker(_posi2, 'Digital');
    _addMarker(_posi3, 'Airport');
  }

  void _addMarker(LatLng position, String markerId) {
    setState(() {
      markers.add(Marker(
          markerId: MarkerId(markerId),
          position: position,
          infoWindow: InfoWindow(title: markerId)));
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("User Current Location"),
        centerTitle: true,
      ),
      body: GoogleMap(

        markers: markers,
        mapType: MapType.hybrid,
        initialCameraPosition: _initialCameraPosition,
        // onMapCreated: (GoogleMapController controller) {
        //   googleMapController = controller;

        // },
        onMapCreated: _onMapCreated,
      ),
      // floatingActionButton: FloatingActionButton.extended(
      //   onPressed: () async {
      //     Position position = await _determinePosition();
      //     googleMapController.animateCamera(CameraUpdate.newCameraPosition(
      //         CameraPosition(
      //             target: LatLng(position.latitude, position.longitude),
      //             zoom: 14)));
      //     markers.clear();
      //     markers.add(Marker(
      //       markerId: const MarkerId("Current Location"),
      //       position: LatLng(position.latitude, position.longitude),
      //     ));
      //     setState(() {});
      //   },
      //   label: const Text("Get Current Location"),
      //   icon: const Icon(Icons.location_history),
      // ),
    );
  }

  Future<Position> _determinePosition() async {
    bool serviceEnabled;
    LocationPermission permission;

    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      return Future.error("Location services are disabled.");
    }

    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        return Future.error("Location permission denied.");
      }
    }

    if (permission == LocationPermission.deniedForever) {
      return Future.error("Location permission permanently denied.");
    }

    return await Geolocator.getCurrentPosition();
  }
}