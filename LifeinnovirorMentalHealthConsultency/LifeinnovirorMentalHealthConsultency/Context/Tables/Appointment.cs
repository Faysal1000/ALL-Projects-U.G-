using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class Appointment
    {
        [Key]
        public int AppointmentId { get; set; }

        [Required]
        public int PatientId { get; set; }
        public Patient Patient { get; set; }

        [Required]
        public int DoctorId { get; set; }
        public Doctor Doctor { get; set; }

        [Required]
        public int SlotId { get; set; }
        public DoctorTimeSlot Slot { get; set; }

        [Required]
        public int AppointmentTypeId { get; set; }
        public AppointmentType AppointmentType { get; set; }

        [Required]
        [StringLength(20)]
        [RegularExpression("Online|Offline", 
            ErrorMessage = "MeetingMedium must be Online or Offline")]
        public string MeetingMedium { get; set; } 

        public string MeetingLink { get; set; }

        public int? PaymentId { get; set; }

        [StringLength(20)]
        [RegularExpression("Booked|Cancelled|Completed", 
            ErrorMessage = "Status must be Booked, Cancelled, or Completed")]
        public string Status { get; set; } = "Booked";

        public string Notes { get; set; }

        public DateTime BookedAt { get; set; } = DateTime.Now;
        public DateTime? CancelledAt { get; set; }
        public DateTime? ConfirmedAt { get; set; }
    }

}